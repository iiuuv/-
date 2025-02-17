from maix import camera, display, image, time, app, touchscreen,nn,uart #从 maix 模块导入多个子模块，包括相机、显示、图像处理、时间、应用、触摸屏和神经网络（nn）模块
import struct

detector = nn.YOLOv5(model="/root/models/model_149737.mud")#yolov5类的实例化，nn模块是用于神经网络模型的加载和执行推理的模块
cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())#创建一个相机对象 cam，并设置其输入宽度、高度和格式为神经网络模型所需的尺寸。
#cam = camera.Camera(224,224)
disp = display.Display()#创建一个显示对象 disp，用于在屏幕上显示图像。
ts = touchscreen.TouchScreen()#创建一个触摸屏对象 ts，用于处理触摸屏输入。
device = "/dev/ttyS0"#定义串行通信设备路径。
serial = uart.UART(device, 115200)#创建一个 UART 对象 serial，用于与串行设备通信，波特率为 115200。

#触摸屏模块，没什么卵用
img = cam.read()#从相机读取一帧图像，并将其存储在 img 变量中。
exit_label = "< Exit"
size = image.string_size(exit_label)
exit_btn_pos = [0, 0, 8*2 + size.width(), 12 * 2 + size.height()]
img.draw_string(8, 12, exit_label, image.COLOR_WHITE)
img.draw_rect(exit_btn_pos[0], exit_btn_pos[1], exit_btn_pos[2], exit_btn_pos[3],  image.COLOR_WHITE, 2)

#颜色阈值
thresholds_red = [[0, 80, 30, 80, 20, 70]]        # red
thresholds_gre = [[0, 80, -50, -20, -10, 40]]      # green
thresholds_blu= [[0, 80, 0, 30, -80, -54]]     # blue25,-30
thresholds = [[58, 70, -2, 8, -14,0]] 

def is_in_button(x, y, btn_pos):
    return x > btn_pos[0] and x < btn_pos[0] + btn_pos[2] and y > btn_pos[1] and y < btn_pos[1] + btn_pos[3]


#串口发送函数，为了实现中心点坐标为(0,0)
def sending_data(B,C,D,E,G):

    D=D-112
    E=E-112
    F=D+E
    if F>127:
        F=F-256
    if F<-128:
        F=F+256 
     
    data_s = struct.pack("<bbbbbb",
                  #(A), 
                   (B), 
                   (C),
                   (E),
                   (D),
                   (F),
                   (G))
    serial.write(data_s)


# #识别二维码函数
# def find_QRcodes():
#     img = cam.read()
    # qrcodes = img.find_qrcodes()
    # for q in qrcodes:
    #     corners = q.corners()
    #     for i in range(4):
    #         img.draw_line(corners[i][0], corners[i][1], corners[(i + 1) % 4][0], corners[(i + 1) % 4][1], image.COLOR_RED)
    #         img.draw_string(0, 20, q.payload(), image.COLOR_BLUE,scale=7)
    #     #serial.write_str(q.payload())
    #     while 1:
    #         sending_data(-10,int(q.payload()[0]),int(q.payload()[1]),int(q.payload()[2]),int(q.payload()[4]),int(q.payload()[5]),int(q.payload()[6]),-9)
    #         #print((int(q.payload()[0]),int(q.payload()[1]),int(q.payload()[2]),int(q.payload()[4]),int(q.payload()[5]),int(q.payload()[6])))
    #         time.sleep_ms(50)
            #disp.show(img)

#我们的车没有陀螺仪，试过巡线的方案，但因为很吃场地，最后没有做到
def find_line():

    img = cam.read()#调用carema中的read方法

    lines = img.get_regression(thresholds, area_threshold = 100)
    for a in lines:
        img.draw_line(a.x1(), a.y1(), a.x2(), a.y2(), image.COLOR_GREEN, 2)
        theta = a.theta()
        rho = a.rho()
        if theta >= 90:
            theta = theta-180
        # else:
        #     theta =  theta
       
        if theta>-128 and theta<127:
            sending_data(-12,theta,theta,theta,-11)#模式1，有红块，中心坐标
        #print(theta)
        time.sleep_ms(1)
        img.draw_string(0, 0, "theta: " + str(theta) + ", rho: " + str(rho), image.COLOR_BLUE)

    disp.show(img)

#识别物料函数
def find_woliao():
    time.sleep_ms(10)
    img = cam.read()
    #红色物料
    blobs1 = img.find_blobs(thresholds_red, area_threshold = 1000, pixels_threshold = 1000)
    for blob in blobs1:
        img.draw_rect(blob[0], blob[1], blob[2], blob[3], image.COLOR_GREEN)
        center_obj1 = ((blob[0] + (blob[2]//2)), (blob[1] + (blob[3]//2)))
        img.draw_string(center_obj1[0],center_obj1[1],"x:" + str(center_obj1[0])+"y:" +str(center_obj1[1]))
        sending_data(-12,0x01,center_obj1[0],center_obj1[1],-11)#模式1，有红块，中心坐标
        print(center_obj1,'red')
        time.sleep_ms(10)
    disp.show(img)
    #绿色物料
    blobs2 = img.find_blobs(thresholds_gre, area_threshold = 1000, pixels_threshold = 1000)
    for blob in blobs2:
        img.draw_rect(blob[0], blob[1], blob[2], blob[3], image.COLOR_GREEN)
        center_obj1 = ((blob[0] + (blob[2]//2)), (blob[1] + (blob[3]//2)))
        img.draw_string(center_obj1[0],center_obj1[1],"x:" + str(center_obj1[0])+"y:" +str(center_obj1[1]))
        sending_data(-12,0x02,center_obj1[0],center_obj1[1],-11)#模式1，有红块，中心坐标
        print(center_obj1,'GREEN')
        time.sleep_ms(10)
    disp.show(img)
    #绿色物料
    blobs3 = img.find_blobs(thresholds_blu, area_threshold = 1000, pixels_threshold = 1000)
    for blob in blobs3:
        img.draw_rect(blob[0], blob[1], blob[2], blob[3], image.COLOR_GREEN)
        center_obj1 = ((blob[0] + (blob[2]//2)), (blob[1] + (blob[3]//2)))
        img.draw_string(center_obj1[0],center_obj1[1],"x:" + str(center_obj1[0])+"y:" +str(center_obj1[1]))
        sending_data(-12,0x03,center_obj1[0],center_obj1[1],-11)#模式1，有红块，中心坐标
        print(center_obj1,'blue')
        time.sleep_ms(10)
    disp.show(img)


#识别圆环函数（粗）
def find_yuanhuan_cu():

    img = cam.read()
    objs = detector.detect(img, conf_th = 0.40, iou_th = 0.45)
    if(objs==[]):
        sending_data(-12,0x04,0,0,-11)

        time.sleep_ms(10)
    for obj in objs:
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = image.COLOR_RED)
        msg = f'{detector.labels[obj.class_id]}: {obj.score:.2f}'
        img.draw_string(obj.x, obj.y, msg, color = image.COLOR_RED)
        xx=obj.x+obj.w//2
        yy=obj.y+obj.h//2
        if obj.class_id==0:
            sending_data(-12,0x02,xx,yy,-11)
            #sending_data(-1,-1,-1,-1,-1)
            img.draw_string(obj.x+obj.w//2,obj.y+obj.h//2,"x:" + str(obj.x+obj.w//2)+"y:" +str(obj.y+obj.h//2))
            #print((obj.x+obj.w//2,obj.y+obj.h//2),'green')
            print(yy-112,xx-112)
            time.sleep_ms(1)
        if obj.class_id==1:#0是绿色，1是蓝色，2是红色
            sending_data(-12,0x04,0,0,-11)
            img.draw_string(obj.x+obj.w//2,obj.y+obj.h//2,"x:" + str(obj.x+obj.w//2)+"y:" +str(obj.y+obj.h//2))
            #print((obj.x+obj.w//2,obj.y+obj.h//2),'blue')
            time.sleep_ms(10)
        if obj.class_id==2:#0是绿色，1是蓝色，2是红色
            sending_data(-12,0x04,0,0,-11)
            img.draw_string(obj.x+obj.w//2,obj.y+obj.h//2,"x:" + str(obj.x+obj.w//2)+"y:" +str(obj.y+obj.h//2))
            #print((obj.x+obj.w//2,obj.y+obj.h//2),'red')
            time.sleep_ms(10)    
    disp.show(img)

#识别圆环函数（细）
def find_yuanhuan_xi():

    img = cam.read()
    gray_img = img.to_format(image.Format.FMT_GRAYSCALE)	
    circles = gray_img.find_circles(threshold = 10000)
    for a in circles:
        gray_img = img.to_format(image.Format.FMT_GRAYSCALE)
        img.draw_circle(a.x(), a.y(), a.r(), image.COLOR_RED, 2)
        img.draw_string(a.x(), a.y(),"x:" + str(a.x())+"y:"+str(a.y()))
        #img.draw_string(a.x() + a.r() + 5, a.y() + a.r() + 5, "r: " + str(a.r()) + "magnitude: " + str(a.magnitude()), image.COLOR_RED)
        sending_data(-12,0x00,a.x(), a.y()-7,-11)
        time.sleep_ms(1) 
        print(a.y()-112, a.x()-112)
    disp.show(img)

def main():
    count = 30
    while not app.need_exit():
        x, y, pressed = ts.read()
        if is_in_button(x, y, exit_btn_pos):
            app.set_exit_flag(True)
      
        img.draw_string(8, 12, exit_label, image.COLOR_WHITE)
        img.draw_rect(exit_btn_pos[0], exit_btn_pos[1], exit_btn_pos[2], exit_btn_pos[3],  image.COLOR_WHITE, 2)
        img.draw_circle(x, y, 1, image.Color.from_rgb(255, 255, 255), 2)
        data = serial.read()
        #根据接受的16进制数据来运行相应模式，
        if data:
            print(data)
            time.sleep_ms(1)
            #模式1，巡线返回角度
            if data == b'\x6C':
                for a in range (0,count):
                    find_line()
            #模式2，寻找物料
            if data == b'\x6D': 
                for a in range (0,count):   
                    find_woliao()
            #模式3，圆环粗定位
            if data == b'\x73':    
                for a in range (0,count):
                    find_yuanhuan_cu()
            #模式4，圆环细定位
            if data == b'\x72':    
                for a in range (0,count):
                    find_yuanhuan_xi()    

if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        e = traceback.format_exc()
        print(e)
        img = image.Image(disp.width(), disp.height())
        img.draw_string(2, 2, e, image.COLOR_WHITE, font="hershey_complex_small", scale=0.6)
        disp.show(img)
        while not app.need_exit():
            time.sleep(0.2)
