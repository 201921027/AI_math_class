import numpy as np

class DiscStat:
    def get_avg(self, list_data): #평균
        sum=0
        for td in list_data:
            sum+=float(td)
        return sum/len(list_data)

    def get_std(self, list_data):  #표준편차
        avg= self.get_avg(list_data)
        ssum=0
        for td in list_data:
            ssum += float(td)*float(td)
        lambda_test= lambda x, list_d: np.sqrt(x/len(list_d))
        return (lambda_test(ssum, list_data))

class TestStat:
    name= "Test Statistics"
    def t_test(self):
        return "t_test"
    def get_name(self):
        return self.name

if __name__=="__main__": #아!! 이게 얘가 홈그라운드에 있을 때만 돌아가는 거구나!! import 해서 불러온 곳에서는 이 함수 안 돌아감
    list_data=[100, 200, 300]
    test0= TestStat()
    print(test0.get_name(), test0.t_test())
    da= DiscStat()
    print("DiscStat()의 get_std(list_data): {:.2f}\n--------------------------".format(da.get_std(list_data)), end="\n\n")


myList=['100', '200', '300']
test= DiscStat()
print("DiscStat()의 get_std(myList): %.2f"%(test.get_std(myList)))