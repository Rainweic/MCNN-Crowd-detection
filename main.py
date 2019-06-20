import argparse

from src import utils


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path",
        type = str,
        default = "./test.jpg",
        help = "图片路径"
    ),
    parser.add_argument(
        "--save_name",
        type = str,
        default = "./",
        help = "保存文件名称(带路径)"
    ),
    parser.add_argument(
        "--show_heatmap",
        type = bool,
        default = False,
        help = "是否显示热力图(仅用于测试效果)"
    )
    parser.add_argument(
        "--is_save",
        type = bool,
        default = False,
        help = "是否保存热力图(未开发)"
    ),
    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help = "端口"
    )
    parser.add_argument(
        "--run_http",
        type = bool,
        default = False,
        help = "运行http服务接受图片"
    ),
    parser.add_argument(
        "--video_path",
        type = str,
        help = "测试视频路径"
    ),
    parser.add_argument(
        "--show_original",
        type = bool,
        default = False,
        help = "生成的demo视频中是否显示原视频"
    )
    args = parser.parse_args()

    return args

def main():
    args = getargs()
    
    if args.run_http:
        utils.run_http(args.port)
    elif args.show_heatmap:
        utils.predict_img(args)
    elif args.video_path:
        utils.predict_video(args)
    else:
        print("没反应? python3 main.py -h 试试")

if __name__ == "__main__":
    main()
    
    
    
