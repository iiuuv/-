#from _typeshed import ReadableBuffer
from maix import camera, display, image, time, app, touchscreen,nn
from maix import uart
import struct

detector = nn.YOLOv5(model="/root/models/model_149737.mud")
cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())
disp = display.Display()
ts = touchscreen.TouchScreen()
device = "/dev/ttyS0"
serial = uart.UART(device, 115200)

thresholds_red = [[0, 80, 20, 60, 10, 40]]        # red
thresholds_gre = [[0, 80, -40, -10, 5, 25]]      # green
thresholds_blu= [[0, 80, -10, 20, -40, -20]]     # blue
thresholds = [[65, 75, -5, 5, -7, 3]] 

#串口发送函数
def sending_data(A,B,C,D,E,G):

    #frame=[0x2C,18,cx%0xff,int(cx/0xff),cy%0xff,int(cy/0xff),0x5B];
    #data = bytearray(frame)
    #F=B+C+D+E#校验和
    data_s = struct.pack("<BBBBBB",      #格式为俩个字符俩个短整型(2字节)
                                         #帧头1
                                         #帧头2
                   (A), # up sample by 4
                   (B), # up sample by 4
                   (C),
                   (D),
                   (E),
                   #(F),
                   (G))
    serial.write(data_s)

# #识别二维码函数
# def find_QRcodes():
#     while 1:
#         img = cam.read()
#         qrcodes = img.find_qrcodes()
#         for qr in qrcodes:
#             corners = qr.corners()
#             for i in range(4):
#                 img.draw_line(corners[i][0], corners[i][1], corners[(i + 1) % 4][0], corners[(i + 1) % 4][1], image.COLOR_RED)
#             img.draw_string(qr.x(), qr.y() - 15, qr.payload(), image.COLOR_RED)
#             print(qr.payload())
#             serial = uart.UART(device, 115200)
#             # sending_data(qr.payload(),0xC2,0xC2,0xC2)
#             serial.write_str(qr.payload())
#             time.sleep_ms(1)
#         disp.show(img)

def find_line():

    
    img = cam.read()

    lines = img.get_regression(thresholds, area_threshold = 100)
    for a in lines:
        img.draw_line(a.x1(), a.y1(), a.x2(), a.y2(), image.COLOR_GREEN, 2)
        theta = a.theta()
        rho = a.rho()
        if theta > 90:
            theta = 270 - theta
        else:
            theta = 90 - theta
        sending_data(0xFA,0xFA,0xFA,0xFB,theta,0xDF)#模式1，有红块，中心坐标
        print(theta)
        time.sleep_ms(1)
        img.draw_string(0, 0, "theta: " + str(theta) + ", rho: " + str(rho), image.COLOR_BLUE)

    disp.show(img)

#识别物料函数
def find_woliao():

    img = cam.read()
    #红色物料
    blobs1 = img.find_blobs(thresholds_red, pixels_threshold=500)
    for blob in blobs1:
        img.draw_rect(blob[0], blob[1], blob[2], blob[3], image.COLOR_GREEN)
        center_obj1 = ((blob[0] + (blob[2]//2)), (blob[1] + (blob[3]//2)))
        sending_data(0xF4,0xA0,0xB1,center_obj1[0],center_obj1[1],0xF5)#模式1，有红块，中心坐标
        print(center_obj1,'red')
        time.sleep_ms(1)
    disp.show(img)
    #绿色物料
    blobs2 = img.find_blobs(thresholds_gre, pixels_threshold=500)
    for blob in blobs2:
        img.draw_rect(blob[0], blob[1], blob[2], blob[3], image.COLOR_GREEN)
        center_obj1 = ((blob[0] + (blob[2]//2)), (blob[1] + (blob[3]//2)))
        sending_data(0xF4,0xB0,0xB1,center_obj1[0],center_obj1[1],0xF5)#模式1，有红块，中心坐标
        print(center_obj1,'GREEN')
        time.sleep_ms(1)
    disp.show(img)
    #绿色物料
    blobs3 = img.find_blobs(thresholds_blu, pixels_threshold=500)
    for blob in blobs3:
        img.draw_rect(blob[0], blob[1], blob[2], blob[3], image.COLOR_GREEN)
        center_obj1 = ((blob[0] + (blob[2]//2)), (blob[1] + (blob[3]//2)))
        sending_data(0xF4,0xC0,0xB1,center_obj1[0],center_obj1[1],0xF5)#模式1，有红块，中心坐标
        print(center_obj1,'blue')
        time.sleep_ms(1)
    disp.show(img)


#识别圆环函数
def find_yuanhuan():

    img = cam.read()
    objs = detector.detect(img, conf_th = 0.7, iou_th = 0.45)
    for obj in objs:
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = image.COLOR_RED)
        msg = f'{detector.labels[obj.class_id]}: {obj.score:.2f}'
        img.draw_string(obj.x, obj.y, msg, color = image.COLOR_RED)
        if obj.class_id==0:
            sending_data(0xF4,0xA0,0xB1,obj.x+obj.w//2,obj.y+obj.h//2,0xF5)
            print((obj.x+obj.w//2,obj.y+obj.h//2),'green')
            time.sleep_ms(1)
        if obj.class_id==1:#0是绿色，1是蓝色，2是红色
            sending_data(0xF4,0xB0,0xB1,obj.x+obj.w//2,obj.y+obj.h//2,0xF5)
            print((obj.x+obj.w//2,obj.y+obj.h//2),'blue')
            time.sleep_ms(1)
        if obj.class_id==2:#0是绿色，1是蓝色，2是红色
            sending_data(0xF4,0xC0,0xB1,obj.x+obj.w//2,obj.y+obj.h//2,0xF5)
            print((obj.x+obj.w//2,obj.y+obj.h//2),'red')
            time.sleep_ms(1)    
    disp.show(img)


def main():
    while not app.need_exit():
        data = serial.read()
        if data:
            print(data)
            time.sleep_ms(1)
            #模式1，巡线返回角度
            if data == b'\x6C':
                find_line()
            #模式2，寻找物料
            if data == b'\x6D':    
                find_woliao()
            #模式3，圆环定位
            if data == b'\x72':    
                find_yuanhuan()

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