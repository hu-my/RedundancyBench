"""
评估 telecom one_one 结果与真值的对比：
1. 原有功能：步骤级别的精确匹配（对比具体冗余步骤索引）
2. 新增功能：轨迹级别的冗余判断（GT 和 LLM 都判断有/无冗余即算对，步骤不需对应）
pred 结果使用 trajectory_index 作为 key，与 ground truth 数组索引对应
"""

import json


def load_ground_truth(path):
    """加载标注文件，返回以数组索引为 key 的 ground truth 字典"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt = {}
    for idx, item in enumerate(data):
        tid = str(item['task_id'])
        gt[idx] = {
            'task_id': tid,
            'redundant_step_idx': set(item.get('redundant_step_idx', []))
        }
    return gt


def load_predictions(path):
    """加载预测结果，返回以 trajectory_index 为 key 的预测字典"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pred = {}
    for item in data:
        idx = item['trajectory_index']
        pred_indices = set()
        for jm in item.get('judged_messages', []):
            pr = jm.get('parsed_result')
            if pr and isinstance(pr, dict) and pr.get('is_redundant') is True:
                pred_indices.add(jm['message_index'])
        pred[idx] = pred_indices
    return pred


def evaluate(gt, pred):
    results = []
    total_tp = total_fp = total_fn = 0
    correct_count = 0

    # redundancy detection counters
    redundancy_match_count = 0
    both_redundant = 0
    both_non_redundant = 0
    gt_redundant_only = 0
    pred_redundant_only = 0

    for idx in sorted(gt.keys()):
        g = gt[idx]
        p = pred.get(idx, set())

        g_set = g['redundant_step_idx']

        # --- original step-level evaluation ---
        tp = len(g_set & p)
        fp = len(p - g_set)
        fn = len(g_set - p)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        is_correct = (g_set == p)
        if is_correct:
            correct_count += 1

        # --- new: trajectory-level redundancy detection ---
        gt_has = len(g_set) > 0
        pred_has = len(p) > 0
        redundancy_match = (gt_has == pred_has)
        if redundancy_match:
            redundancy_match_count += 1
            if gt_has:
                both_redundant += 1
            else:
                both_non_redundant += 1
        else:
            if gt_has:
                gt_redundant_only += 1
            else:
                pred_redundant_only += 1

        results.append({
            'index': idx,
            'task_id': g['task_id'],
            'ground_truth': sorted(g_set),
            'predicted': sorted(p),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'correct': is_correct,
            # new fields
            'gt_has_redundancy': gt_has,
            'pred_has_redundancy': pred_has,
            'redundancy_match': redundancy_match
        })

    total_tasks = len(gt)
    accuracy = correct_count / total_tasks if total_tasks > 0 else 0.0
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    redundancy_detection_accuracy = redundancy_match_count / total_tasks if total_tasks > 0 else 0.0

    return {
        'per_task': results,
        'summary': {
            # step-level metrics
            'total_tasks': total_tasks,
            'correct_tasks': correct_count,
            'accuracy': round(accuracy, 4),
            'overall_precision': round(overall_precision, 4),
            'overall_recall': round(overall_recall, 4),
            'overall_f1': round(overall_f1, 4),
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'step_level': total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0,
            # redundancy detection metrics
            'redundancy_detection_accuracy': round(redundancy_detection_accuracy, 4),
            'both_redundant': both_redundant,
            'both_non_redundant': both_non_redundant,
            'gt_redundant_only': gt_redundant_only,
            'pred_redundant_only': pred_redundant_only
        }
    }


def main():
    gt = load_ground_truth('')
    pred = load_predictions('')

    result = evaluate(gt, pred)

    s = result['summary']

    for r in result['per_task']:
        step_status = "正确" if r['correct'] else "错误"
        red_status = "正确" if r['redundancy_match'] else "错误"


    output_file = ''
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nsave to {output_file}")


if __name__ == "__main__":
    main()
