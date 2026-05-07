"""
评估 all_all 结果与真值的对比：
1. 原有功能：步骤级别的精确匹配（对比具体冗余步骤索引）
2. 新增功能：轨迹级别的冗余判断（GT 和 LLM 都判断有/无冗余即算对，步骤不需对应）
"""

import json


def load_ground_truth(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # list of (task_id, set of redundant_step_idx)
    gt = []
    for item in data:
        tid = str(item['task_id'])
        gt.append((tid, set(item.get('redundant_step_idx', []))))
    return gt


def load_predictions(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # list of set of predicted redundant indices
    pred = []
    for item in data:
        pr = item.get('parsed_result')
        pred_indices = set()
        if pr and isinstance(pr, dict):
            # find the key whose value is a list (the redundant indices)
            for k, v in pr.items():
                if k != 'reason' and isinstance(v, list):
                    pred_indices = set(v)
                    break
        pred.append(pred_indices)
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

    for (tid, g), p in zip(gt, pred):
        # --- original step-level evaluation ---
        tp = len(g & p)
        fp = len(p - g)
        fn = len(g - p)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        is_correct = (g == p)
        if is_correct:
            correct_count += 1

        # --- new: trajectory-level redundancy detection ---
        gt_has = len(g) > 0
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
            'task_id': tid,
            'ground_truth': sorted(g),
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
    gt = load_ground_truth('telecom_自动标注.json')
    pred = load_predictions('gpt4o_all_all_telecom_results.json')

    result = evaluate(gt, pred)

    # print summary
    s = result['summary']
    print("=" * 50)
    print("all_all 评估结果（含冗余判断）")
    print("=" * 50)

    print("\n--- 步骤级别精确匹配 ---")
    print(f"总任务数: {s['total_tasks']}")
    print(f"完全正确的任务数: {s['correct_tasks']}")
    print(f"正确率 (Accuracy): {s['accuracy']:.2%}")
    print(f"总体 Precision: {s['overall_precision']:.4f}")
    print(f"总体 Recall: {s['overall_recall']:.4f}")
    print(f"总体 F1: {s['overall_f1']:.4f}")
    print(f"总 TP: {s['total_tp']}, FP: {s['total_fp']}, FN: {s['total_fn']}")
    print(f"  step-level={s['step_level']}")

    print("\n--- 轨迹级别冗余判断 ---")
    print(f"冗余判断正确数: {s['both_redundant'] + s['both_non_redundant']} / {s['total_tasks']}")
    print(f"冗余判断正确率: {s['redundancy_detection_accuracy']:.2%}")
    print(f"  双方都有冗余: {s['both_redundant']}")
    print(f"  双方都无冗余: {s['both_non_redundant']}")
    print(f"  GT有冗余但预测漏掉: {s['gt_redundant_only']}")
    print(f"  GT无冗余但预测误报: {s['pred_redundant_only']}")
    print("=" * 50)

    # print per-task details
    for r in result['per_task']:
        step_status = "正确" if r['correct'] else "错误"
        red_status = "正确" if r['redundancy_match'] else "错误"
        print(f"\nTask {r['task_id']}")
        print(f"  步骤匹配 [{step_status}] | 冗余判断 [{red_status}]")
        print(f"  真值: {r['ground_truth']}")
        print(f"  预测: {r['predicted']}")
        print(f"  TP={r['tp']} FP={r['fp']} FN={r['fn']}  P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.2f}")

    # save to file
    output_file = 'gpt4oevaluate_all_all_telecom_with_redundancy_result.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到 {output_file}")


if __name__ == "__main__":
    main()