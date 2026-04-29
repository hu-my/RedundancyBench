"""
评估 all_all 结果与真值的对比，计算每个 task 的正确率
"""

import json


def load_ground_truth(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # task_id -> set of redundant_step_idx
    gt = {}
    for item in data:
        tid = str(item['task_id'])
        gt[tid] = set(item.get('redundant_step_idx', []))
    return gt


def load_predictions(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # trajectory_index -> set of predicted redundant indices
    pred = {}
    for item in data:
        idx = str(item['trajectory_index'])
        pr = item.get('parsed_result')
        pred_indices = set()
        if pr and isinstance(pr, dict):
            # find the key whose value is a list (the redundant indices)
            for k, v in pr.items():
                if k != 'reason' and isinstance(v, list):
                    pred_indices = set(v)
                    break
        pred[idx] = pred_indices
    return pred


def evaluate(gt, pred):
    results = []
    total_tp = total_fp = total_fn = 0
    correct_count = 0

    for tid in sorted(gt.keys(), key=int):
        g = gt[tid]
        p = pred.get(tid, set())

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
            'correct': is_correct
        })

    total_tasks = len(gt)
    accuracy = correct_count / total_tasks if total_tasks > 0 else 0.0
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    return {
        'per_task': results,
        'summary': {
            'total_tasks': total_tasks,
            'correct_tasks': correct_count,
            'accuracy': round(accuracy, 4),
            'overall_precision': round(overall_precision, 4),
            'overall_recall': round(overall_recall, 4),
            'overall_f1': round(overall_f1, 4),
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn
        }
    }


def main():
    gt = load_ground_truth('20260426_21标注结果示例.json')
    pred = load_predictions('../all_all_airline_results.json')

    result = evaluate(gt, pred)

    # print summary
    s = result['summary']
    print("=" * 50)
    print("all_all 评估结果")
    print("=" * 50)
    print(f"总任务数: {s['total_tasks']}")
    print(f"完全正确的任务数: {s['correct_tasks']}")
    print(f"正确率 (Accuracy): {s['accuracy']:.2%}")
    print(f"总体 Precision: {s['overall_precision']:.4f}")
    print(f"总体 Recall: {s['overall_recall']:.4f}")
    print(f"总体 F1: {s['overall_f1']:.4f}")
    print(f"总 TP: {s['total_tp']}, FP: {s['total_fp']}, FN: {s['total_fn']}")
    print("=" * 50)

    # print per-task details
    for r in result['per_task']:
        status = "正确" if r['correct'] else "错误"
        print(f"\nTask {r['task_id']} [{status}]")
        print(f"  真值: {r['ground_truth']}")
        print(f"  预测: {r['predicted']}")
        print(f"  TP={r['tp']} FP={r['fp']} FN={r['fn']}  P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.2f}")

    # save to file
    output_file = 'evaluate_all_all_result.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到 {output_file}")


if __name__ == "__main__":
    main()