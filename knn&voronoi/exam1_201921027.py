import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np

def euclidian(data1, data2):
    dist= 0
    for i in range(len(data1)):
        dist += (data1[i] - data2[i]) ** 2
    dist= dist ** 0.5
    # # scipy 라이브러리 이용할 경우
    # dist_lib= distance.euclidean(data1, data2)
    # print(dist, dist_lib) #동일한지 확인
    return dist

def manhatten(data1, data2):
    dist=0
    for i in range(len(data1)):
        dist += abs(data1[i] - data2[i])
    # # scipy 라이브러리 이용할 경우
    # dist_lib= distance.cityblock(data1, data2)
    # print(dist, dist_lib) #동일한지 확인
    return dist

def k_neighbor_f(train, sample, k, dist_func): # sample(= test data)는 data 1개(= 좌표 1개)로 상정
    # # sample로부터 가장 가까운 train data를 k개 추출하는 함수
    dist_list= list()
    k_neighbors= list()
    if dist_func=='euclidian': # 사용할 distance function: euclidian
        # print('k=', k, '\tdistance function:', dist_func)
        # # 모든 train data에 대하여 sample과의 거리 계산
        for train_row in train:
            dist= euclidian(train_row[:2], sample) # euclidian func
            dist_list.append([train_row, dist])
        dist_list.sort(key=lambda i: i[-1])
        # print(dist_list, dist_list[-1]) # 제대로 입력, 정렬 되었는지 확인용
        for i in range(k):
            k_neighbors.append(dist_list[i][0])
        # print(k_neighbors)
        return k_neighbors

    elif dist_func=='manhatten': # 사용할 distance function: manhatten
        # print('k=', k, '\tdistance function:', dist_func)
        # 모든 train data에 대하여 sample과의 거리 계산
        for train_row in train:
            dist = manhatten(train_row[:2], sample) # manhatten func
            dist_list.append([train_row, dist])
        dist_list.sort(key=lambda i: i[-1])
        # print(dist_list) # 제대로 입력, 정렬 되었는지 확인용
        for i in range(k):
            k_neighbors.append(dist_list[i][0])
        # print(k_neighbors)
        return k_neighbors

    else:
        print('해당하지 않는 distance function 입니다.')
        return 0

def KNN(train, sample, k, dist_func):
    # k_neighbor_f()를 호출하여, sample로부터 가장 가까운 train data를 k개 반환받음
    neighbors= k_neighbor_f(train, sample, k, dist_func)
    if neighbors == 0: #해당하지 않는 distance function인 경우
        return 0
    nearest_y= [row[-1] for row in neighbors] # neighbors의 class가 들어감
    # print(nearest_y)
    predict= max(nearest_y, key=nearest_y.count) # sample에 대한 predict 결과
    return predict

def yes_no_f(sample): # yes/no를 1/0으로 변환하는 함수
    yes_no = list()
    for item in sample:
        if item[-1] == 'Yes':
            yes_no.append(1)
        else:
            yes_no.append(0)
    return yes_no

def voronoi_diagram(data, sample, name):
    # all_data= data + sample
    # all_data = data 라고 작성하면 포인터 개념이 되는지, all_data의 변경사항이 data에도 적용되는 문제가 발생(왜?)
    # 그래서 일일이 append 해주었음
    all_data= list()
    for item in data:
        all_data.append(item)
    for item in sample:
        all_data.append(item)
    all_data= np.array(all_data)

    # print(data, all_data) # 정상적으로 적용되었는지 확인

    # label에 따라 그래프에서의 마크 지정을 다르게 하기 위해 yes/no를 1/0으로 변환
    sample = np.array(sample)
    yes_no= yes_no_f(data)
    yes_no_sample= yes_no_f(sample)

    # # 그래프 그리기 위해 speed, agility만 추출(좌표)
    data = np.array(data)
    data= data[:, :2].astype('float')  # 왜인지 str로 들어가서 float으로 형변환
    sample= sample[:, :2].astype('float')

    box1 = {'boxstyle': 'square',
            'ec': (0.0, 0.0, 0.0),
            'fc': (1.0, 1.0, 1.0)}

    vor= Voronoi(all_data[:, :2])
    fig= voronoi_plot_2d(vor, show_vertices=False, show_points=False)
    plt.scatter(data[:, 0], data[:, 1], s=30, c=yes_no,
                cmap=plt.cm.get_cmap('rainbow', 2), label='data')
    plt.scatter(sample[:, 0], sample[:, 1], s=60, c=yes_no_sample,
                cmap=plt.cm.get_cmap('rainbow', 2), marker='+', label='sample')
    plt.legend()
    plt.text(7.345, 1.7, ' Red:     \'Yes\'\n Purple: \'No\'', bbox=box1)
    plt.title(name, size=10)
    plt.show()


# # 데이터 생성
speed_agility_draft= [(2.50, 6.00,'No' ), (3.75, 8.00, 'No'), (2.25, 5.50, 'No'), (3.25, 8.25, 'No'), (2.75, 7.50,'No'),
                       (4.50, 5.00, 'No'), (3.50, 5.25, 'No'), (3.00, 3.25, 'No'), (4.00, 4.00, 'No'), (4.25, 3.75, 'No'),
                       (2.00, 2.00, 'No'), (5.00, 2.50, 'No'), (8.25, 8.50, 'No'), (5.75, 8.75, 'Yes'), (4.75, 6.25, 'Yes'),
                       (5.50, 6.75, 'Yes'), (5.25, 9.50, 'Yes'), (7.00, 4.25, 'Yes'), (7.50, 8.00, 'Yes'), (7.25, 5.75, 'Yes')]

# # distance function이 제대로 작동하는지 확인
# euclidian([1,2], [4,0])
# manhatten([1,2], [4,0])

# # 제대로 작동되는지 sample test
# sample= (6.75, 3)
# predict= KNN(speed_agility_draft, sample, 5, 'euclidian')
# print('input:', sample, '\tpredict: ', predict)

'''문제1: sample= [(6.75, 3), (5.34, 6.0), (4.67, 8.4), (7.0, 7.0), (7.8, 5.4)]에 대하여
3NN-Euclidean, 3NN-Manhattan, 5NN-Euclidean, 5NN-Mantattan 구하기'''
sample= [(6.75, 3), (5.34, 6.0), (4.67, 8.4), (7.0, 7.0), (7.8, 5.4)]
k3_Euclidean_predcit=list()
k3_Manhatten_predcit=list()
k5_Euclidean_predcit=list()
k5_Manhatten_predcit=list()
for data in sample:
    k3_Euclidean= KNN(speed_agility_draft, data, 3, 'euclidian')
    # print('k3_e:', k3_Euclidean)
    k3_e= data+ tuple([k3_Euclidean])
    k3_Euclidean_predcit.append(k3_e)

    k3_Manhatten = KNN(speed_agility_draft, data, 3, 'manhatten')
    # print('k3_m:',  k3_Manhatten)
    k3_m = data + tuple([k3_Manhatten])
    k3_Manhatten_predcit.append(k3_m)

    k5_Euclidean = KNN(speed_agility_draft, data, 5, 'euclidian')
    # print('k5_e:', k5_Euclidean)
    k5_e = data + tuple([k5_Euclidean])
    k5_Euclidean_predcit.append(k5_e)

    k5_Manhatten = KNN(speed_agility_draft, data, 5, 'manhatten')
    # print('k5_m:',  k5_Manhatten)
    k5_m = data + tuple([k5_Manhatten])
    k5_Manhatten_predcit.append(k5_m)

print('\nk= 3, distance function: Euclidean\n' ,k3_Euclidean_predcit)
print('\nk= 3, distance function: Manhatten\n' ,k3_Manhatten_predcit)
print('\nk= 5, distance function: Euclidean\n' ,k5_Euclidean_predcit)
print('\nk= 5, distance function: Manhatten\n' ,k5_Manhatten_predcit)


''' 문제2: 상기 4개 버전에 5개 데이터 모두 포함하여 Voronoi tessellation을 구하고, 
각각에 대한 유사점, 차이점을 논의'''
voronoi_diagram(speed_agility_draft, k3_Euclidean_predcit, '3nn Euclidean')
voronoi_diagram(speed_agility_draft, k3_Manhatten_predcit, '3nn Manhatten')
voronoi_diagram(speed_agility_draft, k5_Euclidean_predcit, '5nn Euclidean')
voronoi_diagram(speed_agility_draft, k5_Manhatten_predcit, '5nn Manhatten')

