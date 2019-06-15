import json
import base64
import time
import cv2
from tornado.options import define
import tornado.web

from src import utils

class MainHandler(tornado.web.RequestHandler):

    def __init__(self):
        self.net = utils.load_model()

    # get方法
    def get(self):
        self.write("运行正常")

    # post方法
    def post(self):
        '''
        request.body读取到的是二进制字符串 
        需要用decode("utf8")转为字符串
        再用json.loads()转为json对象
        '''  
        # 指定摄像头id
        post_id = self.get_argument("id")
        # 是否初始化该摄像头
        is_init = self.get_argument("init")
        # 是否关闭摄像头
        is_close = self.get_argument("close")
        # 从json中获取数据                 
        json_data_byte = self.request.body
        json_data_str = json_data_byte.decode('utf8')
        json_data_obj = json.loads(json_data_str)

        fps = int(json_data_obj.get("Fps"))
        sample = int(json_data_obj.get("SamplingRate"))
        width = int(json_data_obj.get("Width"))
        height = int(json_data_obj.get("Height"))
        image = self.base64_to_image(json_data_obj.get("ImageData"))

        print(fps, sample, width, height)
        print("id", post_id, "is_close", is_close)
        print("is_init", is_init)

        if is_init == "true":
            # 初始化摄像头查询
            pass

        elif is_init == "false":
            if is_close == 'true':
                # 关闭摄像头查询
                pass
            elif is_close == "false":
                t_start = time.time()

                # 处理图片
                img = main.trainsform_img(image)
                density_map = net(img)
                density_map = density_map.data.cpu().numpy()[0][0]
                h, w = density_map.shape
                box_centers = list(zip(density_map.nonzero()[1], density_map.nonzero()[0]))
                count_people = len(box_centers)

                t_end = time.time()

                result = {
                    "count_people": count_people,   # 人群数量估计
                    "box_centers": box_centers,     # 绘制热力图所用坐标
                    "wide_heatmap": w,              # 热力图大小
                    "height_heatmap": h
                }

                print("http：处理图片时间{}".format(t_end-t_start))
                print('writing...')
                print("result--------", result)

                self.write(json.dumps(result))

    def base64_to_image(self, base64_code):
        """
        函数说明:
        转换jpg图片的base64编码为opencv格式
        """
        lens = len(base64_code)
        lenx = lens - (lens % 4 if lens % 4 else 4)

        # try:
        #     img_data = base64.decodestring(strg[:lenx])
        # except:
        #     print("base64_to_image出错")

        # base64解码
        img_data = base64.b64decode(base64_code)
        # 转换为np数组
        img_array = np.fromstring(img_data, np.uint8)
        # 转换成opencv可用格式
        img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    
        return img
        