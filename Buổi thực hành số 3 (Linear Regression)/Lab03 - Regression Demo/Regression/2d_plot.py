### 2차원 그래프 출력 코드
from matplotlib import pyplot as plt

x = [170,155,150,175,165]

y = [65,50,45,70,55]

plt.plot(x,y,'ro')

plt.plot([0,190],[-102.2,83.421])

plt.xlabel('height')

plt.ylabel('weight')

plt.xlim([130,190])

plt.ylim([30,90])

plt.legend(['Samples', 'Linear regression'])

plt.show()