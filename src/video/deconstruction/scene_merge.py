import os
from pathlib import Path
from typing import Optional

from tqdm import tqdm
import json
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src import config

# Lazy import: SentenceTransformer is imported inside __init__ because
# it triggers a HuggingFace model download/check that can hang
# in restricted network environments.
_SentenceTransformer = None
def _get_sentence_transformer():
    global _SentenceTransformer
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as _ST
        _SentenceTransformer = _ST
    return _SentenceTransformer
from src.utils.media_utils import natural_sort_key, hhmmss_to_seconds

# --- Model path resolution ---
_MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'

def _resolve_model_path(model_id: str = _MODEL_ID) -> str:
    """
    Resolve the sentence-transformers model path, checking local caches first:
    1. ModelScope cache: ~/.cache/modelscope/hub/sentence-transformers/all-MiniLM-L6-v2
    2. HuggingFace torch cache: ~/.cache/torch/sentence_transformers/all-MiniLM-L6-v2
    3. Fallback to model_id string (sentence-transformers will download automatically)
    """
    home = Path.home()
    # ModelScope cache layout
    modelscope_path = home / '.cache' / 'modelscope' / 'hub' / 'sentence-transformers' / 'all-MiniLM-L6-v2'
    if modelscope_path.exists():
        print(f"✅ [SceneMerge] Found model in ModelScope cache: {modelscope_path}")
        return str(modelscope_path)

    # HuggingFace / torch sentence_transformers cache layout
    hf_path = home / '.cache' / 'torch' / 'sentence_transformers' / 'all-MiniLM-L6-v2'
    if hf_path.exists():
        print(f"✅ [SceneMerge] Found model in HF cache: {hf_path}")
        return str(hf_path)

    # HuggingFace new cache layout (HF_HOME/hub)
    hf_hub_path = home / '.cache' / 'huggingface' / 'hub' / 'models--sentence-transformers--all-MiniLM-L6-v2'
    if hf_hub_path.exists():
        print(f"✅ [SceneMerge] Found model in HF Hub cache: {hf_hub_path}")
        return str(hf_hub_path)

    # Fallback: let sentence-transformers handle download
    print(f"ℹ️  [SceneMerge] No local cache found, will use model_id: {model_id}")
    return model_id


class OptimizedSceneSegmenter:
    def __init__(self, model_name: Optional[str] = None):
        if model_name is None:
            model_name = _resolve_model_path()
        print(f"🚀 [SceneMerge] Loading model ({model_name})...")
        ST = _get_sentence_transformer()
        self.encoder = ST(model_name)
        
        # 缓存字典
        self.tag_embedding_cache = {} 
        
        # === 状态转移矩阵 (保持不变) ===
        self.loc_matrix = {
            frozenset(["interior", "hybrid"]): 0.6,
            frozenset(["exterior", "hybrid"]): 0.6,
            frozenset(["exterior", "space/abstract"]): 0.1,
            frozenset(["interior", "exterior"]): 0.05,
        }
        self.time_matrix = {
            frozenset(["day", "dawn/dusk"]): 0.6,
            frozenset(["night", "dawn/dusk"]): 0.6,
            frozenset(["day", "night"]): 0.1,
            frozenset(["day", "unclear"]): 0.5,
            frozenset(["night", "unclear"]): 0.5,
        }

    def _pre_compute_embeddings(self, shots):
        """
        [新增] 预计算阶段：
        1. 提取所有出现的唯一环境标签，批量编码。
        2. 提取所有 Shot 的光影色调描述，批量编码。
        """
        print("⚡ [SceneMerge] Pre-computing embeddings for efficiency...")
        
        # --- A. 环境标签去重与编码 ---
        unique_tags = set()
        for shot in shots:
            tags = shot.get('spatio_temporal', {}).get('environment_tags', [])
            unique_tags.update(tags)
            
        unique_tags_list = list(unique_tags)
        
        if unique_tags_list:
            print(f"🏷️  [SceneMerge] Encoding {len(unique_tags_list)} unique environment tags...")
            # 批量编码，速度极快
            embeddings = self.encoder.encode(unique_tags_list)
            # 存入字典: {"Tree": array(...), "Wall": array(...)}
            self.tag_embedding_cache = dict(zip(unique_tags_list, embeddings))
        else:
            self.tag_embedding_cache = {}

        # --- B. 光影色调编码 (每个 Shot 一个，对应列表索引) ---
        print(f"👁️  [SceneMerge] Encoding visual descriptions for {len(shots)} shots...")
        visual_descriptions = []
        for shot in shots:
            st = shot.get('spatio_temporal', {})
            # 拼接光影和色调
            desc = f"{st.get('lighting_mood', '')} {st.get('color_palette', '')}".strip()
            # 如果为空，给一个占位符，防止报错
            visual_descriptions.append(desc if desc else "neutral")
            
        # 返回所有 shot 的光影向量列表，索引与 shots 列表对应
        return self.encoder.encode(visual_descriptions)

    def _get_soft_score(self, val_a, val_b, matrix):
        """查表获取软相似度"""
        if not val_a or not val_b: return 0.5
        val_a, val_b = val_a.lower(), val_b.lower()
        if val_a == val_b: return 1.0
        return matrix.get(frozenset([val_a, val_b]), 0.0)

    def _compute_tag_set_similarity_cached(self, tags_a, tags_b):
        """
        [修改] 使用缓存计算标签集合相似度
        """
        if not tags_a or not tags_b:
            return 0.0
            
        # 从缓存中直接获取向量，不再调用 model.encode
        # 使用列表推导式，处理可能的 KeyError (虽然预计算后理论上不会有)
        vecs_a = [self.tag_embedding_cache[t] for t in tags_a if t in self.tag_embedding_cache]
        vecs_b = [self.tag_embedding_cache[t] for t in tags_b if t in self.tag_embedding_cache]
        
        if not vecs_a or not vecs_b:
            return 0.0
            
        # 转换为 numpy 数组
        vecs_a = np.array(vecs_a)
        vecs_b = np.array(vecs_b)
        
        # 计算相似度矩阵
        sim_matrix = cosine_similarity(vecs_a, vecs_b)
        
        # 双向最佳匹配均值
        best_match_a_to_b = np.max(sim_matrix, axis=1)
        best_match_b_to_a = np.max(sim_matrix, axis=0)
        
        score = (np.mean(best_match_a_to_b) + np.mean(best_match_b_to_a)) / 2.0
        return score

    def _is_bridge_shot(self, shot):
        """判断是否为桥接镜头"""
        narrative = shot.get('narrative_analysis', {})
        func = narrative.get('narrative_function', 'Unknown')
        if func in ['Insert', 'Cut-away', 'Reaction', 'Transition']:
            return True
        cinema = shot.get('cinematography', {})
        scale = cinema.get('shot_scale', '')
        if scale in ['Extreme Close-up', 'Close-up']:
            return True
        return False

    def _extract_long_shot_id(self, shot):
        """提取 long shot id：优先结构化字段，缺失时回退解析文件名。"""
        if not isinstance(shot, dict):
            return None

        raw_id = shot.get('long_shot_id')
        if isinstance(raw_id, int):
            return raw_id
        if isinstance(raw_id, str) and raw_id.isdigit():
            return int(raw_id)

        source_filename = shot.get('source_filename', '')
        if not isinstance(source_filename, str) or not source_filename:
            return None

        # 示例: 2097_2127_shot63_sub6.json / 1867_1869.json
        match = re.search(r'_shot(\d+)(?:_sub\d+)?(?:\.json)?$', source_filename)
        if match:
            return int(match.group(1))
        return None

    def _is_same_long_shot(self, shot_a, shot_b):
        """判断两个相邻片段是否来自同一个 long shot。"""
        shot_id_a = self._extract_long_shot_id(shot_a)
        shot_id_b = self._extract_long_shot_id(shot_b)
        return shot_id_a is not None and shot_id_b is not None and shot_id_a == shot_id_b

    def calculate_similarity(self, shot_a, shot_b, visual_vec_a, visual_vec_b):
        """
        综合相似度计算
        注意：新增了 visual_vec 参数，直接传入预计算好的光影向量
        """
        st_a = shot_a.get('spatio_temporal', {})
        st_b = shot_b.get('spatio_temporal', {})
        
        # 1. 地点类型 (30%)
        sim_loc = self._get_soft_score(st_a.get('location_type'), st_b.get('location_type'), self.loc_matrix)
        
        # 2. 时间状态 (15%)
        sim_time = self._get_soft_score(st_a.get('time_state'), st_b.get('time_state'), self.time_matrix)
        
        # 3. 环境标签集合 (35%) - 使用缓存版本
        sim_env = self._compute_tag_set_similarity_cached(
            st_a.get('environment_tags', []), 
            st_b.get('environment_tags', [])
        )
        
        # 4. 光影与色调 (20%) - 直接计算余弦相似度
        # reshape(1, -1) 是因为 cosine_similarity 需要二维数组
        sim_visual = cosine_similarity(visual_vec_a.reshape(1, -1), visual_vec_b.reshape(1, -1))[0][0]
        sim_visual = max(0, sim_visual)

        # 加权求和
        final_score = (0.30 * sim_loc) + (0.15 * sim_time) + (0.35 * sim_env) + (0.20 * sim_visual)
        return final_score

    def _post_process_merge(self, scenes):
        """
        双阶段后处理：
        1. Forward Pass: 处理短的 Establishment (前奏合并)
        2. Backward Pass: 处理短的 Progression (碎片回收)
        """
        if len(scenes) <= 1:
            return scenes

        # ==================================================
        # Pass 1: Forward Merge (处理 Establishment)
        # 逻辑：短的建立场景 -> 合并到下一个场景的头部
        # ==================================================
        scenes_pass_1 = []
        buffer_shots = []

        print("⚙️  [SceneMerge] Post-processing Pass 1: Merging Establishment shots forward...")
        
        for i in range(len(scenes)):
            current_scene_shots = scenes[i]
            
            # 判断 Establishment 场景
            is_short = len(current_scene_shots) <= 3
            has_establishment = any(
                s.get('narrative_analysis', {}).get('narrative_function') == 'Establishment' 
                for s in current_scene_shots
            )

            if is_short and has_establishment and i < len(scenes) - 1:
                # 放入缓冲区，等待下一个宿主
                buffer_shots.extend(current_scene_shots)
            else:
                # 这是一个宿主场景，接收缓冲区里的镜头
                if buffer_shots:
                    new_scene = buffer_shots + current_scene_shots
                    scenes_pass_1.append(new_scene)
                    buffer_shots = []
                else:
                    scenes_pass_1.append(current_scene_shots)

        # 处理遗留 buffer (合并到最后一个)
        if buffer_shots:
            if scenes_pass_1:
                scenes_pass_1[-1].extend(buffer_shots)
            else:
                scenes_pass_1.append(buffer_shots)

        # ==================================================
        # Pass 2: Backward Merge (处理 Progression) - 新增需求
        # 逻辑：短的 Progression 场景 -> 合并到上一个场景的尾部
        # ==================================================
        if not scenes_pass_1:
            return []

        scenes_pass_2 = []
        print("⚙️  [SceneMerge] Post-processing Pass 2: Merging Progression shots backward...")
        
        # 先放入第一个场景 (因为它前面没有场景可供合并)
        scenes_pass_2.append(scenes_pass_1[0])

        for i in range(1, len(scenes_pass_1)):
            current_scene_shots = scenes_pass_1[i]
            prev_scene_shots = scenes_pass_2[-1] # 获取已处理列表中的最后一个场景

            # 1. 检查长度条件 (<= 5)
            if len(current_scene_shots) <= 5:
                # 2. 统计 Progression 镜头的比例
                progression_count = sum(
                    1 for s in current_scene_shots
                    if s.get('narrative_analysis', {}).get('narrative_function') == 'Progression'
                )
                
                # 3. 判定逻辑: 超过一半是 Progression，或者全部是 Progression
                total = len(current_scene_shots)
                is_mostly_progression = (progression_count / total >= 0.5) if total > 0 else False

                if is_mostly_progression:
                    # === 执行向后合并 ===
                    # 将当前场景追加到上一个场景中
                    # 注意：这里修改的是 scenes_pass_2[-1] 的引用，直接生效
                    prev_scene_shots.extend(current_scene_shots)
                    # 不将 current_scene_shots 作为新条目 append 到 scenes_pass_2
                else:
                    # 正常场景，独立存在
                    scenes_pass_2.append(current_scene_shots)
            else:
                # 长场景，独立存在
                scenes_pass_2.append(current_scene_shots)

        return scenes_pass_2

    def _split_long_scenes(self, scenes, max_duration_secs=300):
        """
        拆分超过最大时长的场景

        Args:
            scenes: 场景列表，每个场景是一个 shot 列表
            max_duration_secs: 单个场景允许的最大时长（秒），默认5分钟

        Returns:
            拆分后的场景列表
        """
        if not scenes or max_duration_secs <= 0:
            return scenes

        split_scenes = []
        scenes_split_count = 0

        for scene_shots in scenes:
            if not scene_shots:
                continue

            # 计算场景时长
            first_shot = scene_shots[0]
            last_shot = scene_shots[-1]

            start_time = first_shot.get('duration', {}).get('clip_start_time', 0)
            end_time = last_shot.get('duration', {}).get('clip_end_time', 0)

            # 如果时间信息是字符串格式，尝试转换
            if isinstance(start_time, str):
                start_time = hhmmss_to_seconds(start_time)
            if isinstance(end_time, str):
                end_time = hhmmss_to_seconds(end_time)

            scene_duration = end_time - start_time

            if scene_duration <= max_duration_secs:
                # 场景时长在限制内，保持不变
                split_scenes.append(scene_shots)
            else:
                # 场景时长超过限制，需要拆分
                scenes_split_count += 1

                # 计算需要拆分成多少段
                num_segments = int(np.ceil(scene_duration / max_duration_secs))
                target_duration_per_segment = scene_duration / num_segments

                current_segment = []
                segment_start_time = start_time

                for idx, shot in enumerate(scene_shots):
                    shot_end_time = shot.get('duration', {}).get('clip_end_time', 0)
                    if isinstance(shot_end_time, str):
                        shot_end_time = hhmmss_to_seconds(shot_end_time)

                    current_segment.append(shot)

                    # 检查当前段是否已达到目标时长
                    segment_duration = shot_end_time - segment_start_time

                    if segment_duration >= target_duration_per_segment and len(current_segment) >= 1:
                        # 硬约束：同一个 long shot 的相邻 sub-clip 不允许被拆到不同 scene
                        if idx + 1 < len(scene_shots):
                            next_shot = scene_shots[idx + 1]
                            if self._is_same_long_shot(shot, next_shot):
                                continue

                        # 保存当前段，开始新段
                        split_scenes.append(current_segment)
                        current_segment = []
                        segment_start_time = shot_end_time

                # 处理剩余的 shots
                if current_segment:
                    # 如果剩余部分太短，合并到前一个段
                    if split_scenes and len(current_segment) <= 2:
                        split_scenes[-1].extend(current_segment)
                    else:
                        split_scenes.append(current_segment)

        if scenes_split_count > 0:
            original_count = len(scenes)
            new_count = len(split_scenes)
            print(f"[Scene Split] Split {scenes_split_count} long scenes (>{max_duration_secs}s) "
                  f"into smaller segments: {original_count} -> {new_count} scenes")

        return split_scenes

    def _scene_duration_secs(self, scene_shots):
        """计算单个 scene 的时长（秒）。"""
        if not scene_shots:
            return 0.0

        start_time = scene_shots[0].get('duration', {}).get('clip_start_time', 0)
        end_time = scene_shots[-1].get('duration', {}).get('clip_end_time', 0)

        if isinstance(start_time, str):
            start_time = hhmmss_to_seconds(start_time)
        if isinstance(end_time, str):
            end_time = hhmmss_to_seconds(end_time)

        try:
            return max(0.0, float(end_time) - float(start_time))
        except Exception:
            return 0.0

    def _merge_tiny_duration_scenes(self, scenes, min_scene_duration_secs=2.0):
        """将极短 scene 合并到相邻 scene：首段并后，其余并前。"""
        if not scenes or min_scene_duration_secs <= 0:
            return scenes

        merged = []
        pending_front = []

        for idx, scene_shots in enumerate(scenes):
            if not scene_shots:
                continue

            duration = self._scene_duration_secs(scene_shots)
            is_tiny = duration < min_scene_duration_secs

            if is_tiny:
                if not merged and idx < len(scenes) - 1:
                    # 第一个 scene 太短，缓冲后并入下一个 scene
                    pending_front.extend(scene_shots)
                elif merged:
                    # 非首段太短，直接并入前一个 scene
                    merged[-1].extend(scene_shots)
                else:
                    merged.append(scene_shots)
                continue

            if pending_front:
                merged.append(pending_front + scene_shots)
                pending_front = []
            else:
                merged.append(scene_shots)

        if pending_front:
            if merged:
                merged[-1].extend(pending_front)
            else:
                merged.append(pending_front)

        tiny_count = sum(1 for s in scenes if self._scene_duration_secs(s) < min_scene_duration_secs)
        if tiny_count > 0:
            print(f"🧲 [SceneMerge] Absorbed {tiny_count} tiny scenes (<{min_scene_duration_secs}s)")

        return merged


    def segment(self, shots, threshold=0.5, max_scene_duration_secs=300):
        if not shots:
            return []
            
        # === 步骤 1: 执行预计算 ===
        # visual_vectors 是一个列表，visual_vectors[i] 对应 shots[i] 的光影向量
        visual_vectors = self._pre_compute_embeddings(shots)
        
        scenes = []
        current_scene = [shots[0]]
        
        print("🔍 [SceneMerge] Segmenting scenes with optimized similarity...")
        i = 1
        with tqdm(total=len(shots)-1) as pbar:
            while i < len(shots):
                prev_shot = shots[i-1]
                curr_shot = shots[i]
                
                # 调用相似度计算，传入对应的预计算向量
                sim_score = self.calculate_similarity(
                    prev_shot, curr_shot,
                    visual_vectors[i-1], visual_vectors[i]
                )
                
                is_cut = sim_score < threshold

                # 硬约束：同一个 long shot 的相邻 sub-clip 之间禁止切 scene
                if is_cut and self._is_same_long_shot(prev_shot, curr_shot):
                    is_cut = False
                
                # === 叙事平滑 (Look-Ahead) ===
                if is_cut and self._is_bridge_shot(curr_shot):
                    if i + 1 < len(shots):
                        next_shot = shots[i+1]
                        # 跨越比较
                        bridge_sim = self.calculate_similarity(
                            prev_shot, next_shot,
                            visual_vectors[i-1], visual_vectors[i+1]
                        )
                        if bridge_sim >= threshold:
                            is_cut = False
                # ===========================
                
                if is_cut:
                    scenes.append(current_scene)
                    current_scene = [curr_shot]
                else:
                    current_scene.append(curr_shot)
                
                i += 1
                pbar.update(1)
            
        scenes.append(current_scene)
        # 3. 执行后处理合并
        merged_scenes = self._post_process_merge(scenes)

        # 4. 拆分超长场景
        final_scenes = self._split_long_scenes(merged_scenes, max_scene_duration_secs)

        # 5. 吸收极短 scene（如 1s 插入镜头），避免碎片化
        min_scene_duration = getattr(config, 'MIN_SCENE_DURATION_SECS', 0.0)
        final_scenes = self._merge_tiny_duration_scenes(final_scenes, min_scene_duration)
        return final_scenes


def load_shots(input_dir):
    """读取并排序所有 Shot JSON"""
    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    files.sort(key=natural_sort_key)
    
    shots_data = []
    print(f"📂 [SceneMerge] Loading {len(files)} shots from {input_dir}...")
    
    for f in files:
        path = os.path.join(input_dir, f)
        try:
            with open(path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                # 我们可以把文件名也存进去，方便溯源
                data['source_filename'] = f 
                shots_data.append(data)
        except Exception as e:
            print(f"❌ [SceneMerge] Error loading {f}: {e}")
            
    return shots_data

def save_scenes(scenes, output_dir):
    """保存 Scene JSON，并计算聚合时间"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"💾 [SceneMerge] Saving {len(scenes)} merged scenes to {output_dir}...")
    
    for idx, scene_shots in enumerate(scenes):
        if not scene_shots:
            continue
            
        # === 聚合 Scene 元信息 ===
        first_shot = scene_shots[0]
        last_shot = scene_shots[-1]
        
        # 获取起止时间 (假设 duration 字段存在且格式正确)
        # 如果第一步生成的 JSON 里是字符串格式的 '00:00:10'，这里可能需要转换
        # 这里假设之前代码已经将其存为了 float 秒数
        start_time = first_shot.get('duration', {}).get('clip_start_time', 0)
        end_time = last_shot.get('duration', {}).get('clip_end_time', 0)
        
        scene_metadata = {
            "scene_id": idx,
            "shot_count": len(scene_shots),
            "time_range": {
                "start_seconds": start_time,
                "end_seconds": end_time,
            },
            "frame_range": [first_shot.get('frame_range', [0,0])[0], last_shot.get('frame_range', [0,0])[1]],
            "shot_list": [s.get('source_filename', 'unknown') for s in scene_shots],
            # 将所有原始 shot 数据包含在内，供后续 LLM 归纳使用
            "shots_data": scene_shots 
        }
        
        # 写入文件 scene_0.json, scene_1.json ...
        output_path = os.path.join(output_dir, f"scene_{idx}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scene_metadata, f, indent=2, ensure_ascii=False)

def main():
    # 1. 加载数据
    print("📂 [SceneMerge] Loading shots...")
    shots = load_shots(SHOTS_DIR)
    print(f"📄 [SceneMerge] Total shots loaded: {len(shots)}")
    
    if not shots:
        print("⚠️  [SceneMerge] No shots found. Exiting.")
        return

    # 2. 初始化分割器
    segmenter = OptimizedSceneSegmenter()
    
    # 3. 执行切分
    # threshold 建议：
    # 0.4 - 0.5: 倾向于合并 (场景更长，容错率高)
    # 0.6 - 0.7: 倾向于切分 (场景更碎，更敏感)
    # max_scene_duration_secs: 超过此时长（秒）的场景将被拆分
    merged_scenes = segmenter.segment(
        shots,
        threshold=config.SCENE_SIMILARITY_THRESHOLD,
        max_scene_duration_secs=config.MAX_SCENE_DURATION_SECS
    )
    
    print(f"🧩 [SceneMerge] Merged {len(shots)} shots into {len(merged_scenes)} scenes.")
    
    # 4. 保存结果
    save_scenes(merged_scenes, SCENES_DIR)
    print("✅ [SceneMerge] Done!")

if __name__ == "__main__":
    main()