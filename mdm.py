
import math

knownWidth = 1.8
knownHeight = 1.5
focalLength = 2.5
CCD_W = 4.8
CCD_H = 3.6
IMAGE_W = 1280
IMAGE_H = 720
veh_speed = 3.0  # 本车车速 km/h
dis_last = 0.0


class MDM(object):
    # 测速测距处理函数
    def mdm_info_proc(self, index, boxs, cur_s, last_s, dis_last, fcw):
        #print("=======mdm_info========")
        t_s = float(cur_s - last_s)
        #dis_last = 0
        ttc = 0
        dis_x = 0
        dis_y = 0
        z = 0
        bar_rel_spd = 0
        #print("t_s:", t_s)
        #print("id:", index, "box:", int(boxs[0]), int(boxs[1]), int(boxs[2]), int(boxs[3]))
        # z = (knownWidth * focalLength) / box[2] #距离相机垂直距离
        # center_x = box[0] + float(box[2] / 2) #图像坐标系矩形框底边中心x
        # center_y = box[1] + box[3] #图像坐标系矩形框底边中心y
        # z = (knownWidth * focalLength) / (boxs[2] - boxs[0])  # 距离相机垂直距离

        # 1、像素坐标系转换为图像坐标系
        pixel_w = boxs[2] - boxs[0]
        pixel_h = boxs[3] - boxs[1]

        dx = float(CCD_W / IMAGE_W) #单个像素代表的宽度 mm
        dy = float(CCD_H / IMAGE_H) ##单个像素代表的高度 mm

        pixel_u = boxs[0] + float((boxs[2] - boxs[0]) / 2)  # 矩形框底边中心x
        pixel_v = boxs[3]  # 矩形框底边中心y

        img_x = float((pixel_u - IMAGE_W/2) * dx)
        img_y = float((pixel_v - IMAGE_H/2) * dy)

        # 2、图像坐标系转相机坐标系
        z = float((knownHeight * focalLength) / (pixel_h * dy))  # 距离相机垂直距离
        cam_x = float(z / focalLength * img_x)  # 左负右正

        # 3、相机坐标系下车辆距离
        dis_y = z  # 纵向距离
        dis_x = cam_x  # 横向距离

        if t_s != 0 and dis_last[index-1] != 0:
            if abs(dis_x) < (0.9 + 0.5):  # 判断cipv，车身往外0.5米范围内
                bar_speed = float((dis_y - dis_last[index-1]) / t_s)  # 前车速度 m/s
                bar_rel_spd = bar_speed - veh_speed * 10 / 36  # 相对速度=前-自
                if bar_rel_spd != 0:
                    ttc = float(dis_y / bar_rel_spd)
                if ttc < 2.6 and dis_y< 3.5:
                    fcw[0] = 1
                    #print("FCW-前车碰撞预警")

        dis_last[index-1] = dis_y  #y应该取上一帧记录的值

        #print("dis_y:", dis_y, "dis_x:", dis_x, "m", "ttc:", ttc, "rel_spd:", bar_rel_spd, "m/s")
        return z