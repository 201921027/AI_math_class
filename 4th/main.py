import numpy as np
import statpy

print("------------------------------\n")
# 남자 키 평균 및 표준편차 출력 코드
fp= open("data_IIM.csv", 'r')
fpw= open("data_test_w.txt", 'w')
man_list= list()

for line in fp:
    line= line.strip("\r\n")
    psd= line.split(',')    #리스트로 반환

    if int(psd[1])==1:   #남자면 append
        man_list.append(psd[2])
        print("{} append( height: {} )".format(psd[0], psd[2]))
        fpw.write("{} append( height: {} )\n".format(psd[0], psd[2]))
    else:   #여자면 continue
        continue

fp.close()
print("man_list: ", man_list)
print("------------------------------\n")
man_height= statpy.DiscStat()
man_avg= man_height.get_avg(man_list)
man_std= man_height.get_std(man_list)

print("man_avg: {:.2f}\nman_std: {:.2f}".format(man_avg, man_std))
fpw.write("\nman_avg: {:.2f}\nman_std: {:.2f}".format(man_avg, man_std))
fpw.close()




