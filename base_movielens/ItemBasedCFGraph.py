import matplotlib.pyplot as plt
import fileinput

plt.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体
#x轴的点
x = [5, 10, 20, 40, 80,160]
#y轴的点
#精度
# y1 = [16.96, 17.23, 17.26, 16.59, 16.11, 16.24]
# #回调率
# y2 = [19.81, 20.12, 20.16, 19.37, 18.81, 18.96]
# #覆盖率
# y3 = [18.93,15.34,13.12,11.92,11.56,12.10]
# #流行度
# y4 = [5.539260,5.619290,5.624129,5.617081,5.625827,5.632205]
y1 = []
y2 = []
y3 = []
y4 = []
#读取result_ibcf.data 中的数据到列表
for line in fileinput.input("result_ibcf.data"):
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
plt.title('ItemBasedCF')
#x坐标描述标签
plt.xlabel('K value')
#y坐标描述标签
plt.ylabel('percent(%)')

plt.plot(x, y1, '-o',label='precision')
plt.plot(x, y2,'-or', label='recall')
plt.plot(x, y3, '-og',label='Coverage')
plt.plot(x, y4,'-oy', label='Popularity')
plt.legend()
plt.savefig('ItemBasedCF.png')
#显示
plt.show()