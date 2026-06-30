from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging
import os
import gc

from omegaconf import DictConfig
from hydra.utils import instantiate

from nuplan.planning.training.experiments.cache_metadata_entry import (
    CacheMetadataEntry,
    CacheResult,
    save_cache_metadata,
)

from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from navsim.common.dataloader import SceneLoader, SceneFilter
from navsim.common.dataclasses import SensorConfig, Scene

logger = logging.getLogger(__name__)


def cache_single_scenario(
    scene_dict: Dict[str, Any], 
    processor: MetricCacheProcessor,
    cfg: DictConfig
) -> Optional[CacheMetadataEntry]:
    """
    单独处理单个场景的缓存逻辑【核心函数，可直接断点】
    :param scene_dict: 单场景的原始数据字典
    :param processor: 缓存处理器
    :param cfg: 配置文件
    :return: 缓存元数据，失败返回None
    """
    # 构建Scene对象
    scene = Scene.from_scene_dict_list(
        scene_dict,
        None,
        num_history_frames=cfg.train_test_split.scene_filter.num_history_frames,
        num_future_frames=cfg.train_test_split.scene_filter.num_future_frames,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    # 构建NavSimScenario数据集对象【重点：这里可以断点看scenario的所有属性】
    scenario = NavSimScenario(
        scene, 
        map_root=os.environ["NUPLAN_MAPS_ROOT"], 
        map_version="nuplan-maps-v1.0"
    )
    # 核心缓存计算逻辑【断点在这里可以看计算后的缓存结果】
    return processor.compute_metric_cache(scenario)


def cache_data(cfg: DictConfig) -> None:
    """
    【单进程主入口】纯for循环串行执行缓存逻辑，完全无多进程
    :param cfg: omegaconf 全局配置字典
    """
    # 校验缓存路径必填
    assert cfg.cache.cache_path is not None, f"Cache path cannot be None when caching, got {cfg.cache.cache_path}"
    cache_path = Path(cfg.cache.cache_path)
    # 临时添加，为了确保地址符合格式
    cfg.navsim_log_path = cfg.navsim_log_path.replace('navsim_logs', 'meta_datas') 
    
    navsim_log_path = Path(cfg.navsim_log_path)
    
    # 1. 初始化场景加载器、过滤器，加载需要缓存的场景数据
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    # 获取需要处理的日志文件+token列表（和原逻辑一致）
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    logger.info("Starting metric caching (单进程串行) of %s log files...", str(len(data_points)))

    # 2. 初始化缓存处理器【全局唯一，复用】
    processor = MetricCacheProcessor(
        cache_path=cache_path,
        force_feature_computation=cfg.cache.force_feature_computation,
    )

    # 3. 初始化统计变量 + 缓存元数据列表
    total_successes = 0  # 成功缓存的场景数
    total_failures = 0   # 失败的场景数
    all_cache_metadata: List[CacheMetadataEntry] = []  # 所有成功的缓存元数据

    # ==========================================
    # ✅ 核心：纯for循环 串行遍历所有日志文件
    # ==========================================
    for data_idx, data_point in enumerate(data_points):
        current_log_file = data_point["log_file"]
        current_tokens = data_point["tokens"]
        current_cfg = data_point["cfg"]
        logger.info(f"\n===== 处理第 {data_idx+1}/{len(data_points)} 个日志文件: {current_log_file} =====")
        
        # 为当前日志文件重新初始化加载器（过滤指定log和token）
        single_log_filter = instantiate(cfg.train_test_split.scene_filter)
        single_log_filter.log_names = [current_log_file]
        single_log_filter.tokens = current_tokens
        single_log_loader = SceneLoader(
            sensor_blobs_path=None,
            data_path=navsim_log_path,
            scene_filter=single_log_filter,
            sensor_config=SensorConfig.build_no_sensors(),
        )
        scene_count = len(single_log_loader.scene_frames_dicts)
        logger.info(f"当前日志文件包含 {scene_count} 个待缓存场景")

        # ==========================================
        # ✅ 核心：纯for循环 串行遍历当前日志的所有场景
        # ==========================================
        for scene_idx, scene_dict in enumerate(single_log_loader.scene_frames_dicts.values()):
            logger.info(f"处理场景: {scene_idx+1}/{scene_count} | log_file={current_log_file}")
            
            # 处理单个场景【⭐⭐⭐ 最佳断点位置：这里能一步步看每个场景的处理过程】
            cache_meta = cache_single_scenario(scene_dict, processor, current_cfg)
            
            # 更新统计
            if cache_meta is not None:
                total_successes += 1
                all_cache_metadata.append(cache_meta)
            else:
                total_failures += 1
            
            # 强制垃圾回收，防止内存泄漏（处理大场景必备，原逻辑保留）
            gc.collect()

    # 4. 打印最终统计结果
    total_scenarios = total_successes + total_failures
    if total_failures == 0:
        logger.info(
            "✅ 完成全量缓存！所有 %s 个场景缓存成功.", str(total_scenarios)
        )
    else:
        logger.info(
            "⚠️ 完成缓存！成功: %s | 失败: %s | 总计: %s",
            str(total_successes), str(total_failures), str(total_scenarios)
        )

    # 5. 保存缓存元数据csv文件（和原逻辑一致，单节点固定node_id=0）
    node_id = 0
    logger.info(f"保存缓存元数据文件到: {cache_path}")
    save_cache_metadata(all_cache_metadata, cache_path, node_id)
    logger.info("✅ 元数据文件保存完成！")