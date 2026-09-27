"""docRenderCut 启动入口。

    python app.py                          # 数据默认在 <项目>/data，监听 127.0.0.1:8848
    python app.py --data D:/drc-data       # 数据放外部 / 共享目录
    python app.py --port 8849              # 同机多实例用不同端口

数据目录优先级：--data  >  环境变量 DOCRENDERCUT_DATA  >  <项目>/data
生产部署常把数据放在共享盘，多实例读写同一份；ID 分配已用 mkdir 原子占位，
并发创建分类 / 模板 / 任务不会撞号。
"""

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="docRenderCut",
        description="业务单据标准模板渲染与模板块裁剪服务",
    )
    ap.add_argument(
        "--data", default=None, metavar="DIR",
        help="数据目录（默认 <项目>/data；也可用环境变量 DOCRENDERCUT_DATA）",
    )
    ap.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    ap.add_argument("--port", type=int, default=8848, help="监听端口（默认 8848）")
    args = ap.parse_args()

    # 必须在导入 app.main 之前落到环境变量：store.py 在导入时解析数据目录
    if args.data:
        data_path = Path(args.data).expanduser().resolve()
        os.environ["DOCRENDERCUT_DATA"] = str(data_path)
        os.environ["DOCRENDERCUT_DATA_SOURCE"] = "cli"
        try:
            data_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"[docRenderCut] 数据目录不可用：{data_path}（{exc}）", file=sys.stderr)
            return 2
        print(f"[docRenderCut] 数据目录：{data_path}（来自 --data）")
    elif os.environ.get("DOCRENDERCUT_DATA", "").strip():
        print(f"[docRenderCut] 数据目录：{os.environ['DOCRENDERCUT_DATA']}（来自环境变量）")
    else:
        # 不做隐式默认：数据目录未配置时，前端会引导完成"配置 + 探测校验"。
        # 这样项目目录里永远不会冒出 data/。
        print("[docRenderCut] 数据目录未配置 —— 打开页面后会引导完成初始化")

    print(f"[docRenderCut] 启动 http://{args.host}:{args.port}")

    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
