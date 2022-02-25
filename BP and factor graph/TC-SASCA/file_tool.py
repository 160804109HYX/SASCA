import numpy as np
s = [0, 1, 2, 3, 8188, 8189, 8190, 8191] # candidates of  key value

def double_s(bw1_one,Numberoftripple):
    array = np.array([0] * (Numberoftripple[0]*Numberoftripple[0]))
    for i in range(Numberoftripple[0]):
        for j in range(Numberoftripple[0]):
            array[i * Numberoftripple[0] + j] = (bw1_one[i] + bw1_one[j])%65536
    return array


def four_s(bw1_one,Numberoftripple):
    data = np.array([0] * (Numberoftripple[0]*Numberoftripple[0]*Numberoftripple[0]*Numberoftripple[0]))
    for i in range(Numberoftripple[0]):
        for j in range(Numberoftripple[0]):
            for k in range(Numberoftripple[0]):
                for x in range(Numberoftripple[0]):
                    data[i * Numberoftripple[0] * Numberoftripple[0] * Numberoftripple[0] + j * Numberoftripple[0] * Numberoftripple[0] + k * Numberoftripple[0] + x] = (bw1_one[i] + bw1_one[j] + bw1_one[k] + bw1_one[x])%65536
    return data

def aw2(bw1_one,bw2_one,Numberoftripple):
    temp_two = np.array([[0] * Numberoftripple] * (8*8*8*8))
    sum_two = np.array([0] * (8*8*8*8))
    for i in range(8):
        for j in range(8):
            for k in range(8):
                for x in range(8):
                    sum_two[i * 8*8*8 + j * 8*8 + k * 8 + x] = (8*bw1_one[i] + 4*bw1_one[j] + 2*bw1_one[k] + bw1_one[x])%65536
    for i in range((8*8*8*8)):
        for j in range(Numberoftripple):
            if sum_two[i] == bw2_one[j]:
                t = j
        temp_two[i, t] = 1
    temp_two = temp_two.reshape((8,8,8,8, Numberoftripple))

    return temp_two

def aw3(bw1_one,bw2_one,Numberoftripple):
    temp_two = np.array([[0] * Numberoftripple] * (8*8*8*8))
    sum_two = np.array([0] * (8*8*8*8))
    for i in range(8):
        for j in range(8):
            for k in range(8):
                for x in range(8):
                    sum_two[i * 8*8*8 + j * 8*8 + k * 8 + x] = (bw1_one[i] + bw1_one[j] + bw1_one[k] + bw1_one[x])%65536
    for i in range((8*8*8*8)):
        for j in range(Numberoftripple):
            if sum_two[i] == bw2_one[j]:
                t = j
        temp_two[i, t] = 1
    temp_two = temp_two.reshape((8,8,8,8, Numberoftripple))

    return temp_two

def aw4(bw1_one,bw2_one,Numberoftripple):
    temp_two = np.array([[0] * Numberoftripple] * (8*8*8*8))
    sum_two = np.array([0] * (8*8*8*8))
    for i in range(8):
        for j in range(8):
            for k in range(8):
                for x in range(8):
                    sum_two[i * 8*8*8 + j * 8*8 + k * 8 + x] = (-bw1_one[i] + bw1_one[j] -bw1_one[k] + bw1_one[x])%65536
    for i in range((8*8*8*8)):
        for j in range(Numberoftripple):
            if sum_two[i] == bw2_one[j]:
                t = j
        temp_two[i, t] = 1
    temp_two = temp_two.reshape((8,8,8,8, Numberoftripple))

    return temp_two

def aw5(bw1_one,bw2_one,Numberoftripple):
    temp_two = np.array([[0] * Numberoftripple] * (8*8*8*8))
    sum_two = np.array([0] * (8*8*8*8))
    for i in range(8):
        for j in range(8):
            for k in range(8):
                for x in range(8):
                    sum_two[i * 8*8*8 + j * 8*8 + k * 8 + x] = (bw1_one[i] + 2*bw1_one[j] + 4*bw1_one[k] + 8*bw1_one[x])%65536
    for i in range((8*8*8*8)):
        for j in range(Numberoftripple):
            if sum_two[i] == bw2_one[j]:
                t = j
        temp_two[i, t] = 1
    temp_two = temp_two.reshape((8,8,8,8, Numberoftripple))

    return temp_two

def aw6(bw1_one,bw2_one,Numberoftripple):
    temp_two = np.array([[0] * Numberoftripple] * (8*8*8*8))
    sum_two = np.array([0] * (8*8*8*8))
    for i in range(8):
        for j in range(8):
            for k in range(8):
                for x in range(8):
                    sum_two[i * 8*8*8 + j * 8*8 + k * 8 + x] = (-bw1_one[i] + 2*bw1_one[j] - 4*bw1_one[k] + 8*bw1_one[x])%65536
    for i in range((8*8*8*8)):
        for j in range(Numberoftripple):
            if sum_two[i] == bw2_one[j]:
                t = j
        temp_two[i, t] = 1
    temp_two = temp_two.reshape((8,8,8,8, Numberoftripple))

    return temp_two

def overflowing_mul(x, y):
    mul = x * y
    mul = mul & 0x0000FFFF
    return mul


def hm_weight(n):
    count = 0
    while n != 0:
        count += n & 0x1
        n >>= 1
    return count


def slice_probability(path_probability):
    probability = np.loadtxt(path_probability)
    in_probability = probability.reshape((100, 144, 17))
    temp = in_probability
    return temp

def gen_single_matrix(Numberoftripple,b,x,bw1_one,bw1_two,bw1_four,flag):
    data = []
    single_index=[0,1,3,4]
    double_index=[2,5,6,7]
    for k in range(4):
        for i in range(16):
            temp = np.array([[0] * 17] * Numberoftripple[0])   #17
            for j in range(Numberoftripple[0]):
                t = hm_weight(overflowing_mul(bw1_one[j], b[x][i][single_index[k]]))
                temp[j, t] = 1
            data.append(temp)

    for k in range(4):
        for i in range(16):
            temp = np.array([[0] * 17] * Numberoftripple[1])
            for j in range(Numberoftripple[1]):
                t = hm_weight(overflowing_mul(bw1_two[j], b[x][i][double_index[k]]))
                temp[j, t] = 1
            data.append(temp)
    if flag==1:
        for i in range(16):
            temp = np.array([[0] * 17] * Numberoftripple[2])
            for j in range(Numberoftripple[2]):
                t = hm_weight(overflowing_mul(bw1_four[j], b[x][i][8]))
                temp[j, t] = 1
            data.append(temp)
    return data

def gen_single_matrix_1(b,x,tar):
    data = []
    for i in range(16):
        temp = np.array([[0] * 17] * 8)   #17
        for j in range(8):
            t = hm_weight(overflowing_mul(s[j], b[x][i][tar]))
            temp[j, t] = 1
        data.append(temp)
    return data

def gen_base_matrix(bw1_one,bw1_two,bw1_four,Numberoftripple,flag):
    data = []

    temp_two = np.array([[0] * Numberoftripple[1]] * (Numberoftripple[0]*Numberoftripple[0]))
    sum_two = double_s(bw1_one,Numberoftripple)
    for i in range((Numberoftripple[0]*Numberoftripple[0])):
        for j in range(Numberoftripple[1]):
            if sum_two[i] == bw1_two[j]:
                t = j
        temp_two[i, t] = 1
    temp_two = temp_two.reshape((Numberoftripple[0], Numberoftripple[0], Numberoftripple[1]))
    data.append(temp_two)
    if(flag==1):
        temp_four = np.array([[0] * Numberoftripple[2]] * (Numberoftripple[0]*Numberoftripple[0]*Numberoftripple[0]*Numberoftripple[0]))
        sum_four = four_s(bw1_one,Numberoftripple)
        for i in range((Numberoftripple[0]*Numberoftripple[0]*Numberoftripple[0]*Numberoftripple[0])):
            for j in range( Numberoftripple[2]):
                if sum_four[i] == bw1_four[j]:
                    t = j
            temp_four[i, t] = 1
        temp_four = temp_four.reshape((Numberoftripple[0], Numberoftripple[0], Numberoftripple[0], Numberoftripple[0], Numberoftripple[2]))
        data.append(temp_four)

    return data

def gen_base_whole(bw1_one,bw2_one,bw3_one,bw4_one,bw5_one,bw6_one,Numberoftripple):
    data = []

    data2 = aw2(bw1_one, bw2_one, Numberoftripple[1])
    data3 = aw3(bw1_one, bw3_one, Numberoftripple[2])
    data4 = aw4(bw1_one, bw4_one, Numberoftripple[3])
    data5 = aw5(bw1_one, bw5_one, Numberoftripple[4])
    data6 = aw6(bw1_one, bw6_one, Numberoftripple[5])
    data.append(data2)
    data.append(data3)
    data.append(data4)
    data.append(data5)
    data.append(data6)
    return data

def slice_probkey(fgtype,noiselevel,n,filepath,flag):
    probekeyfile=probekeyfile=filepath+"sim-template-30-GS"+str(noiselevel)+"-randk/"
    f = open(probekeyfile+"Aw"+fgtype+"_prob.txt", "r")
    for i in range(n):
        with open(probekeyfile+'data_single.txt', 'a') as f1:
            f1.write(f.readline())
            f1.write(f.readline())
            f1.write(f.readline())
            f1.write(f.readline())

        with open(probekeyfile+'data_double.txt', 'a') as f2:
            f2.write(f.readline())
            f2.write(f.readline())
            f2.write(f.readline())
            f2.write(f.readline())
        if flag==1:
            with open(probekeyfile+'data_four.txt', 'a') as f3:
                f3.write(f.readline())

def string_to_float(str):
    return float(str)




