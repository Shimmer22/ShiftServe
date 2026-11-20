"""身份字段供主机准确率实验的执行器和逐题结果共用。"""

import hashlib
import json


def run_metadata(protocol, cfg, condition, limit_tasks, task_ids, runtime):
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    meta = dict(protocol_version=protocol, config_sha256=digest(cfg),
                condition=condition, limit_tasks=limit_tasks, task_ids=task_ids, runtime=runtime)
    return dict(meta, run_sha256=digest(meta))


def same_run(record, meta):
    return all(record.get(key) == value for key, value in meta.items())


def select_tasks(tasks, id_for, ids, limit):
    if ids:
        wanted = set(ids)
        tasks = [task for task in tasks if id_for(task) in wanted]
        if {id_for(task) for task in tasks} != wanted:
            raise ValueError("请求的题目不在当前评测题集")
    if limit:
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError("没有可运行的评测题目")
    return tasks
