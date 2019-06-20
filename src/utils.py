import cv2
import numpy as np
import os
import torch
import time
from pyheatmap.heatmap import HeatMap
from .http_server import MainHandler
import tornado.web
import tornado.ioloop
from . import network
from .crowd_count import CrowdCounter

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# 模型路径
model_path = './final_models/mcnn_shtechA_660.h5'

def load_model():
    '''
    加载模型
    '''
    net = CrowdCounter()
    trained_model = os.path.join(model_path)
    network.load_net(trained_model, net)
    net.cuda()
    net.eval()
    return net

def trainsform_img(img):
    '''
    transform图片
    '''
    h, w = img.shape[0], img.shape[1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (w, h))
    img = img.reshape((1, 1, img.shape[0], img.shape[1]))

    return img

def predict_img(args):
    '''
    对图片进行预测
    '''
    net = load_model()

    img_with_c3 = cv2.imread(args.img_path)
    # 转为gray
    img = trainsform_img(img_with_c3)
    density_map = net(img)
    density_map = density_map.data.cpu().numpy()

    deal_density_map(args, img_with_c3, density_map)

def predict_video(args, FPS=60):
    '''
    对视频进行预测
    '''
    net = load_model()

    cap = cv2.VideoCapture(args.video_path)
    w, h = int(cap.get(3)),int(cap.get(4))
    if args.show_original:
        h *= 2
    save_cap = cv2.VideoWriter(args.save_name, cv2.VideoWriter_fourcc('I', '4', '2', '0'), FPS, (w,h))
    while(cap.isOpened()):
        start = time.time()
        ret, frame = cap.read()
        image = trainsform_img(frame)
        density_map = net(image)
        density_map = density_map.data.cpu().numpy()[0][0]
        
        box_centers = list(zip(density_map.nonzero()[1], density_map.nonzero()[0]))
        heatmap_img = density_heatmap(density_map.shape[1], density_map.shape[0], box_centers)
        heatmap_img = cv2.resize(heatmap_img, (frame.shape[1], frame.shape[0]))
        image_with_heatmap = cv2.addWeighted(frame, 0.7, heatmap_img, 0.3, 0)
        end = time.time()
        print("处理一个帧所需时间(不计网络传输) {}".format(end-start))	

        if args.show_original:
            image_with_heatmap = np.vstack((frame, image_with_heatmap))

        if args.show_heatmap:
            cv2.imshow("video frame", image_with_heatmap)
            cv2.waitKey(1)

        save_cap.write(image_with_heatmap)

    cap.release()
    save_cap.release()
    cv2.destroyAllWindows()
        

def deal_density_map(args, img_with_c3, density_map):
    '''
    处理热力图 显示、保存等功能
    '''
    if args.show_heatmap:
        display_heatmap(img_with_c3, density_map)
    if args.is_save:
        pass

def density_heatmap(width, height, box_centers, r=4):
    '''
    制作热力图
    '''
    hm = HeatMap(box_centers, width=width, height=height)
    heatmap_img = hm.heatmap(r=r)
    # 将PIL转opencv
    heatmap_img = cv2.cvtColor(np.asarray(heatmap_img),cv2.COLOR_RGB2BGR)
    return heatmap_img

def display_heatmap(img_with_c3, density_map):
    '''
    仅显示热力图
    img_with_c3: 具有3通道的色彩图片
    density_map: 预测后的密度map。
    '''
    density_map = density_map[0][0]
    
    box_centers = list(zip(density_map.nonzero()[1], density_map.nonzero()[0]))
    # print(density_map.shape)
    # print(box_centers)
    heat_map = density_heatmap(density_map.shape[1], density_map.shape[0], box_centers)
 
    # density_map的大小与圆图不一样 被缩小 故生成的heat_map要缩放到原来大小
    heat_map = cv2.resize(heat_map, (img_with_c3.shape[1], img_with_c3.shape[0]))
    
    img = cv2.addWeighted(img_with_c3, 0.7, heat_map, 0.3, 0)

    cv2.imshow("test", img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

def run_http(port):
    settings = {'debug' : True}
    app = tornado.web.Application([
        (r'/crowd', MainHandler)
    ], **settings)
    http_server = tornado.httpserver.HTTPServer(app)
    print("port: {}".format(port))
    http_server.bind(port)
    http_server.start(1)    # 0为多线程 1为单线程
    tornado.ioloop.IOLoop.instance().start()
    


