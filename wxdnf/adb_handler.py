import os
import subprocess
import datetime
#
# for i in range(1):
#
#     order='adb shell screencap /data/screen.png'        #获取连接设备
#     pi= subprocess.Popen(order,shell=True,stdout=subprocess.PIPE)
#
#     print(pi.stdout.read())      #打印结果
#
#     p2=subprocess.Popen('adb pull /data/screen.png /Users/user/Documents/MuMu共享文件夹/screen'+str(i)+'.png',shell=True,stdout=subprocess.PIPE)
#     print(p2.stdout.read())
#
# while True:
#     print(subprocess.Popen("adb shell input tap 178 775",shell=True,stdout=subprocess.PIPE))
#     time.sleep(0.1)
#     #print(subprocess.Popen("adb shell input tap 1285 770",shell=True,stdout=subprocess.PIPE))

def click_handle(x,y):
    return subprocess.Popen("adb shell input tap "+x+" "+y,shell=True,stdout=subprocess.PIPE)

def get_screen(i):
    os.system('adb exec-out screencap -p > /Users/user/Documents/MuMu共享文件夹/screen'+str(i)+'.jpg')

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
for i in range(4):
    pipe = subprocess.Popen("adb exec-out screencap -p",
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, shell=True)
    image_bytes = pipe.stdout.read()
    print(image_bytes)
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
