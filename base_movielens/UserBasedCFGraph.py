import matplotlib.pyplot as plt
import fileinput

plt.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体
#x轴的点
x = [5, 10, 20, 40, 80,160]
#y轴的点
#精度
# y1 = [16.71, 19.40, 21.64, 22.48, 22.05, 21.28]
# #回调率
# y2 = [14.18, 16.45, 18.36, 19.07, 18.70, 18.06]
# #覆盖率
# y3 = [41.99,30.90,24.46,18.01,14.40,11.87]
# #流行度
# y4 = [5.218520,5.360592,5.465238,5.544084,5.606586,5.658175]
y1 = []
y2 = []
y3 = []
y4 = []
#读取result_ubcf.data 中的数据到列表
for line in fileinput.input("result_ubcf.data"):
    t = line.split(' ')
    for i in range(len(t)):
        if(i == 1):
            y1.append(float(t[1]))
        if(i == 2):
            y2.append(float(t[2]))
        if(i == 3):
            y3.append(float(t[3]))
        if(i == 4):
            y4.append(float(t[4]))


#标题
plt.title('UserBasedCF')
#x坐标描述标签
plt.xlabel('K value')
#y坐标描述标签
plt.ylabel('percent(%)')

plt.plot(x, y1, '-o',label='precision')
plt.plot(x, y2,'-or', label='recall')
plt.plot(x, y3, '-og',label='Coverage')
plt.plot(x, y4,'-oy', label='Popularity')
plt.legend()
plt.savefig('UserBasedCF.png')
#显示
plt.show()