# 导入PIL模块
import PIL
# 从PIL模块中导入Image
from PIL import Image

# 使用open()函数打开路径中的图片
# 赋值给变量img
img = Image.open(r"C:\Users\19336\Desktop\QSY.jpg")
# 利用convert函数将图片转化为RGB模式
rgb_img = img.convert('RGB')
# 利用size()方法获取图片的宽度和高度
# 将宽度、高度分别赋值给变量width，height
width, height = rgb_img.size
# for循环遍历图片中每个像素点的像素值
for i in range(width):
    for j in range(height):
        pos = (i, j)
        # TODO 如果一个像素点的三个像素值之和大于等于540，则将像素值赋值为（255，255，255）
        if sum(rgb_img.getpixel(pos)) >= 540:
            rgb_img.putpixel(pos, (255, 255, 255))

# 将图片保存到路径中
rgb_img.save(r"C:\Users\19336\Desktop\QSYH.jpg")
