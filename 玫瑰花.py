import turtle as tl
 
tl.speed(0)
tl.setup(600, 650)		# 设置画面尺寸
tl.title('海龟绘制复杂图案——简笔画漂亮的玫瑰花')
tl.up()
tl.goto(-180, 275)		# 坐标移至(-200,290)
# 绘制文字
tl.color('blue')		# 写标题的颜色(蓝色) 
tl.write('漂亮的玫瑰花',align='center',font=('微软雅黑',24))
R = 1/3				# 设置比例因子，用数据的1/3大小绘制
x0, y0 = R*650, R*850		# 设置玫瑰花中心位置偏移量
# 花瓣背景(花瓣主体)
tl.goto(R*1109-x0,y0-R*308)	# 花瓣背景起点坐标
tl.color('red')			# 设置花瓣为红色(画笔、填充都为红色)
tl.seth(-25); tl.down(); tl.begin_fill()
tl.circle(-400*R,38); tl.rt(115); tl.circle(300*R,45); tl.circle(-700*R,20)
tl.circle(700*R,4); tl.circle(-550*R,10); tl.circle(-300*R,27)
tl.circle(50*R,35); tl.circle(-800*R,5); tl.circle(-60*R,70)
tl.circle(900*R,12); tl.circle(100*R,27); tl.circle(-800*R,14)
tl.circle(-30*R,121); tl.circle(-800*R,15); tl.lt(90); tl.circle(400*R,18)
tl.circle(-50*R,140); tl.lt(130); tl.circle(-130*R,110); tl.circle(100*R,30)
tl.circle(-450*R,20); tl.circle(-250*R,60); tl.lt(70); tl.circle(-130*R,60)
tl.circle(500*R,21); tl.circle(-130*R,95); tl.circle(-200*R,20); tl.end_fill()
# 画花叶
tl.color('green')		# 设置花叶为绿色
tl.up(); tl.goto(R*560-x0,y0-R*1230)				# 花叶起点坐标
tl.down(); tl.seth(80); tl.begin_fill()
for i in range(5):		# 画第一片花叶
    tl.circle(410*R,20); tl.lt(120); tl.fd(35*R); tl.rt(120)	# 画叶缘
tl.fd(160*R); tl.lt(150); tl.fd(150*R)
for i in range(4):		# 画叶缘
    tl.rt(130); tl.fd(30*R); tl.lt(120+i*3); tl.circle(600*R,11-i)
tl.rt(130); tl.fd(30*R); tl.lt(125); tl.circle(500*R,30); tl.end_fill()
tl.up(); tl.seth(-35); tl.fd(150*R); tl.lt(185); tl.down()	# 画叶脉
tl.color('darkgreen')		# 叶脉用深绿色
tl.begin_fill(); tl.circle(-470*R,20); tl.circle(750*R,25); tl.fd(200*R)
tl.lt(175); tl.fd(190*R); tl.circle(-750*R,20); tl.circle(500*R,15)
tl.fd(100*R); tl.end_fill(); tl.up(); tl.lt(175); tl.fd(250*R)
tl.rt(45); tl.down(); tl.pensize(3); tl.fd(160*R)
for i in range(2):
    tl.up(); tl.lt(120); tl.fd(110*R); tl.rt(115); tl.down(); tl.fd((150-i*20)*R)
tl.up(); tl.lt(140); tl.fd(150*R); tl.lt(120); tl.down(); tl.fd(100*R)
for i in range(2):
    tl.up(); tl.rt(110); tl.fd((100+i*5)*R); tl.lt(110); tl.down(); tl.fd(140*R)
 
tl.color('green')		# 设置线条颜色和填充颜色同为绿色
tl.up(); tl.seth(-80)		# 第二片花叶
tl.fd(250*R); tl.pensize(1); tl.down(); tl.seth(125); tl.begin_fill()
for i in range(5):
    tl.circle(330*R,20); tl.lt(120); tl.fd(30*R); tl.rt(120)
tl.fd(130*R); tl.lt(150); tl.fd(120*R)
for i in range(4):
    tl.rt(130); tl.fd(25*R); tl.lt(120+i*3); tl.circle(460*R,11-i)
tl.rt(130); tl.fd(25*R); tl.lt(125); tl.circle(400*R,30); tl.end_fill()
tl.up(); tl.seth(12); tl.fd(120*R); tl.lt(182); tl.down()
tl.color('darkgreen'); tl.begin_fill(); tl.circle(-380*R,20)
tl.circle(600*R,30); tl.fd(130*R); tl.lt(175); tl.fd(120*R)
tl.circle(-600*R,25); tl.circle(400*R,15); tl.fd(80*R); tl.end_fill()
tl.up(); tl.lt(175); tl.fd(200*R); tl.rt(45); tl.down(); tl.pensize(3)
tl.fd(150*R)
for i in range(2):
    tl.up(); tl.lt(120); tl.fd(90*R); tl.rt(115); tl.down(); tl.fd((100-i*15)*R)
tl.up(); tl.lt(140); tl.fd(110*R); tl.lt(120); tl.down(); tl.fd(100*R)
for i in range(2):
    tl.up(); tl.rt(110); tl.fd((80+i*5)*R); tl.lt(110); tl.down(); tl.fd(110*R)
tl.up(); tl.rt(10); tl.fd(500*R); tl.seth(90); tl.color('green'); tl.pensize(1)
tl.begin_fill()			# 画第三片花叶
for i in range(5):
    tl.circle(-410*R,20); tl.rt(120); tl.fd(35*R); tl.lt(120)
tl.fd(160*R); tl.rt(150); tl.fd(150*R)
for i in range(4):
    tl.lt(130); tl.fd(30*R); tl.rt(120+i*3); tl.circle(-600*R,11-i)
tl.lt(130); tl.fd(30*R); tl.rt(125); tl.circle(-500*R,30); tl.end_fill()
tl.up(); tl.lt(40); tl.fd(160*R); tl.lt(180); tl.down(); tl.color('darkgreen')
tl.begin_fill(); tl.circle(470*R,20); tl.circle(-750*R,25); tl.fd(200*R)
tl.rt(175); tl.fd(190*R); tl.circle(750*R,20); tl.circle(-500*R,15)
tl.fd(100*R); tl.end_fill(); tl.up(); tl.rt(175); tl.fd(250*R); tl.lt(45)
tl.down(); tl.pensize(3); tl.fd(160*R)
for i in range(2):
    tl.up(); tl.rt(120); tl.fd(110*R); tl.lt(115); tl.down(); tl.fd((150-i*20)*R)
tl.up(); tl.rt(140); tl.fd(150*R); tl.rt(120); tl.down(); tl.fd(100*R)
for i in range(2):
    tl.up(); tl.lt(110); tl.fd((100+i*5)*R); tl.rt(110); tl.down(); tl.fd(140*R)
tl.lt(45); tl.fd(230*R); tl.lt(45)
tl.color('saddlebrown','#976123')				# 花梗色
tl.begin_fill()			# 画花梗
tl.fd(460*R); tl.circle(250*R,15); tl.rt(120); tl.circle(80*R,45); tl.rt(100)
tl.circle(-100*R,35); tl.circle(700*R,15); tl.fd(360*R); tl.circle(300*R,10)
tl.fd(220*R); tl.rt(90); tl.fd(35*R); tl.rt(90); tl.fd(220*R)
tl.circle(-320*R,10); tl.fd(70*R); tl.end_fill()
tl.up()				# 画花萼
tl.rt(160); tl.fd(400*R); tl.lt(80); tl.pensize(1); tl.color('darkgreen','green')
tl.down(); tl.begin_fill(); tl.circle(520*R,10); tl.circle(100*R,75)
tl.circle(-100*R,60); tl.lt(150); tl.circle(220*R,85); tl.rt(140)
tl.circle(240*R,30); tl.circle(-240*R,15); tl.lt(140); tl.circle(-240*R,20)
tl.circle(240*R,25); tl.rt(120); tl.circle(180*R,80); tl.circle(100*R,40)
tl.lt(100); tl.circle(-310*R,23); tl.lt(57); tl.circle(-520*R,35); tl.end_fill()
tl.up(); tl.seth(90)		# 画花芯花瓣(蕃茄红色)
tl.fd(595*R); tl.lt(90); tl.fd(60*R); tl.rt(120); tl.down(); tl.color('tomato')
tl.begin_fill(); tl.circle(-80*R,45); tl.circle(-30*R,120); tl.circle(-10*R,50)
tl.circle(-250*R,30); tl.circle(-20*R,60); tl.circle(-80*R,100)
tl.circle(-290*R,40); tl.circle(-110*R,90); tl.circle(110*R,70)
tl.circle(-110*R,80); tl.circle(-220*R,25); tl.rt(70); tl.fd(25*R); tl.rt(110)
tl.circle(220*R,30); tl.circle(80*R,70); tl.circle(-120*R,30); tl.lt(125)
tl.circle(200*R,20); tl.rt(45); tl.circle(220*R,50); tl.rt(80); tl.fd(21*R)
tl.rt(105); tl.circle(-400*R,33); tl.lt(50); tl.circle(-240*R,15)
tl.circle(850*R,5); tl.circle(90*R,87); tl.circle(270*R,40); tl.circle(60*R,100)
tl.circle(10*R,63); tl.circle(250*R,25); tl.lt(90); tl.circle(28*R,100)
tl.end_fill(); tl.up(); tl.rt(150); tl.fd(95*R); tl.rt(30); tl.begin_fill()
tl.down(); tl.circle(-160*R,30); tl.circle(-270*R,30); tl.circle(-100*R,50)
tl.rt(150); tl.circle(250*R,50); tl.circle(150*R,20); tl.end_fill(); tl.up()
tl.lt(130); tl.fd(280*R); tl.rt(65); tl.begin_fill(); tl.down()
tl.circle(-350*R,20); tl.circle(-100*R,10); tl.circle(-470*R,10); tl.rt(110)
tl.fd(20*R); tl.rt(60); tl.circle(-470*R,8); tl.circle(150*R,25)
tl.circle(470*R,8); tl.circle(-30*R,50); tl.end_fill(); tl.up(); tl.lt(186,)
tl.fd(480*R); tl.lt(195); tl.down(); tl.begin_fill(); tl.circle(-500*R,22)
tl.circle(280*R,20); tl.circle(130*R,90); tl.circle(15*R,175); tl.fd(20*R)
tl.rt(105); tl.circle(-300*R,70); tl.rt(120); tl.circle(-800*R,3); tl.rt(70)
tl.circle(100*R,30); tl.lt(160); tl.circle(-50*R,75); tl.circle(-100*R,100)
tl.circle(200*R,20); tl.lt(30); tl.fd(25*R); tl.lt(135); tl.fd(35*R); tl.rt(95)
tl.circle(-100*R,80); tl.circle(800*R,15); tl.circle(-100*R,100)
tl.circle(-200*R,30); tl.circle(800*R,20); tl.circle(10*R,180)
tl.circle(-800*R,20); tl.circle(220*R,22); tl.circle(120*R,105)
tl.circle(-800*R,15); tl.circle(110*R,90); tl.rt(85); tl.circle(-400*R,5)
tl.circle(130*R,100); tl.rt(60); tl.circle(600*R,5); tl.rt(50)
tl.circle(600*R,5); tl.circle(40*R,120); tl.circle(600*R,15); tl.lt(50)
tl.circle(-600*R,15); tl.lt(130); tl.circle(170*R,55); tl.lt(165)
tl.circle(-160*R,40); tl.rt(120); tl.fd(90*R); tl.rt(45)
tl.circle(-600*R,14); tl.circle(-30*R,130); tl.circle(500*R,20)
tl.circle(300*R,50); tl.rt(95); tl.circle(-120*R,85); tl.circle(700*R,19)
tl.end_fill(); tl.up(); tl.rt(66); tl.fd(150*R); tl.rt(57); tl.down()
tl.begin_fill(); tl.circle(50*R,50); tl.fd(150*R); tl.circle(600*R,8)
tl.circle(-200*R,10); tl.circle(-30*R,145); tl.circle(-300*R,45)
tl.circle(-350*R,20); tl.circle(500*R,40); tl.circle(-200*R,20)
tl.circle(-50*R,100); tl.circle(-200*R,20); tl.circle(-500*R,20)
tl.circle(-250*R,40); tl.circle(-500*R,20); tl.circle(-700*R,20)
tl.circle(100*R,45); tl.fd(30*R); tl.rt(180); tl.circle(-120*R,45)
tl.circle(800*R,15); tl.circle(500*R,25); tl.circle(275*R,38)
tl.circle(500*R,27); tl.circle(70*R,110); tl.circle(220*R,25)
tl.circle(-420*R,43); tl.circle(320*R,65); tl.circle(55*R,141)
tl.circle(100*R,10); tl.fd(280*R); tl.end_fill(); tl.up(); tl.rt(165)
tl.fd(210*R); tl.lt(20); tl.down(); tl.begin_fill(); tl.circle(700*R,20)
tl.circle(30*R,120); tl.circle(800*R,15); tl.circle(-80*R,30)
tl.circle(-800*R,12); tl.circle(80*R,70); tl.circle(500*R,10); tl.lt(135)
tl.fd(20*R); tl.lt(42); tl.circle(-480*R,8); tl.circle(-70*R,70)
tl.circle(800*R,13); tl.circle(70*R,30); tl.circle(-800*R,13)
tl.circle(-25*R,123); tl.circle(-680*R,17); tl.end_fill(); tl.up(); tl.lt(80)
tl.fd(200*R); tl.lt(20); tl.down(); tl.begin_fill(); tl.circle(-130*R,100)
tl.circle(250*R,20); tl.circle(-400*R,30); tl.circle(-250*R,20)
tl.circle(-200*R,35); tl.rt(90); tl.fd(20*R); tl.rt(90); tl.circle(200*R,35)
tl.circle(230*R,20); tl.circle(380*R,30); tl.circle(-260*R,20)
tl.circle(115*R,105); tl.end_fill(); tl.up(); tl.lt(55); tl.fd(870*R)
tl.rt(25); tl.down(); tl.begin_fill(); tl.circle(-400*R,40); tl.rt(115)
tl.circle(300*R,45); tl.circle(-700*R,19); tl.rt(135); tl.fd(25*R); tl.rt(43)
tl.circle(700*R,18); tl.circle(-300*R,39); tl.lt(113); tl.circle(400*R,35)
tl.end_fill(); tl.lt(20); tl.up(); tl.fd(300*R); tl.lt(25); tl.down()
tl.begin_fill(); tl.circle(-400*R,50); tl.circle(300*R,5); tl.lt(170)
tl.circle(-300*R,6); tl.circle(250*R,25); tl.circle(400*R,36)
tl.circle(15*R,180); tl.end_fill()
 
tl.ht()				# 隐藏turtle形状
tl.done()