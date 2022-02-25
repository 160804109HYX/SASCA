from bp_tools import *
from file_tool import *
from time import *

'''
KFG: Karatsuba Factor Graph
'''
def ComputerBpAw(fgtype,n,noiselevel,filepath,Numberoftripple,data,flag):
    loopnum=3
    probekeyfile=filepath+"sim-template-30-GS"+str(noiselevel)+"-randk/"
    outname = probekeyfile + fgtype+'_attack_result.txt'
    path1_probability = probekeyfile+'data_single.txt'
    path2_probability = probekeyfile+'data_double.txt'
    path3_probability = probekeyfile+'data_four.txt'
    probability1 = np.loadtxt(path1_probability)
    probability2 = np.loadtxt(path2_probability)
    if (flag == 1):
        probability3 = np.loadtxt(path3_probability)
    in_probability1 = probability1.reshape((n, 4, Numberoftripple[0]))
    in_probability2 = probability2.reshape((n, 4, Numberoftripple[1]))
    ck_path = filepath+"ck_index.txt"
    ck_index = np.loadtxt(ck_path, dtype=int)
    count=np.array([0]*5)
    time_count = np.array([0] * n, dtype=float)
    if(flag==1):
        in_probability3 = probability3.reshape((n, Numberoftripple[2]))

    for ff in range(n):
        print(ff)
        begin_time = time()
        mrf = string2factor_graph(
            'f1(a,b,e)f2(a,c,f)f3(b,d,g)f4(c,d,h)f6(a)f7(b)f8(c)f9(d)f10(e)f11(f)f12(g)f13(h)')

        f1 = factor(['a', 'b', 'e'], data[0])
        f2 = factor(['a', 'c', 'f'], data[0])
        f3 = factor(['b', 'd', 'g'], data[0])
        f4 = factor(['c', 'd', 'h'], data[0])

        f6 = factor(['a'], in_probability1[ff, 0])
        f7 = factor(['b'], in_probability1[ff, 1])
        f8 = factor(['c'], in_probability1[ff, 2])
        f9 = factor(['d'], in_probability1[ff, 3])

        f10 = factor(['e'], in_probability2[ff, 0])
        f11 = factor(['f'], in_probability2[ff, 1])
        f12 = factor(['g'], in_probability2[ff, 2])
        f13 = factor(['h'], in_probability2[ff, 3])

        mrf.change_factor_distribution('f1', f1)
        mrf.change_factor_distribution('f2', f2)
        mrf.change_factor_distribution('f3', f3)
        mrf.change_factor_distribution('f4', f4)

        mrf.change_factor_distribution('f6', f6)
        mrf.change_factor_distribution('f7', f7)
        mrf.change_factor_distribution('f8', f8)
        mrf.change_factor_distribution('f9', f9)

        mrf.change_factor_distribution('f10', f10)
        mrf.change_factor_distribution('f11', f11)
        mrf.change_factor_distribution('f12', f12)
        mrf.change_factor_distribution('f13', f13)

        lbp_a = loopy_belief_propagation(mrf)

        t_a = []
        t_b = []
        t_c = []
        t_d = []
        for i in range(loopnum):
            t_a.append(lbp_a.belief('a', i).get_distribution())
            t_b.append(lbp_a.belief('b', i).get_distribution())
            t_c.append(lbp_a.belief('c', i).get_distribution())
            t_d.append(lbp_a.belief('d', i).get_distribution())


        count_a = t_a[loopnum-1]
        count_b = t_b[loopnum-1]
        count_c = t_c[loopnum-1]
        count_d = t_d[loopnum-1]

        if(flag==1):
            mrf = string2factor_graph('f1(a,b,c,d,z)f2(a)f3(b)f4(c)f5(d)f6(z)')

            f1 = factor(['a', 'b', 'c', 'd', 'z'], data[1])
            f2 = factor(['a'], count_a)
            f3 = factor(['b'], count_b)
            f4 = factor(['c'], count_c)
            f5 = factor(['d'], count_d)

            f6 = factor(['z'], in_probability3[ff])

            mrf.change_factor_distribution('f1', f1)
            mrf.change_factor_distribution('f2', f2)
            mrf.change_factor_distribution('f3', f3)
            mrf.change_factor_distribution('f4', f4)
            mrf.change_factor_distribution('f5', f5)
            mrf.change_factor_distribution('f6', f6)

            bp = belief_propagation(mrf)
            count_a = bp.belief('a').get_distribution()
            count_b = bp.belief('b').get_distribution()
            count_c = bp.belief('c').get_distribution()
            count_d = bp.belief('d').get_distribution()

        end_time=time()
        time_count[ff]=end_time-begin_time
        if ck_index[ff,0]==np.argmax(count_a):
            count[0]+=1
        if ck_index[ff,1]==np.argmax(count_b):
            count[1]+=1
        if ck_index[ff,3]==np.argmax(count_c):
            count[2]+=1
        if ck_index[ff,4]==np.argmax(count_d):
            count[3]+=1
        if (ck_index[ff,0]==np.argmax(count_a))&(ck_index[ff,1]==np.argmax(count_b))&(ck_index[ff,3]==np.argmax(count_c))&(ck_index[ff,4]==np.argmax(count_d)):
            count[4]+=1
        with open(outname, 'a') as f:
            np.savetxt(f, (count_a, count_b, count_c, count_d))
        with open(probekeyfile+fgtype+"_maxindex.txt", 'a') as f:
            f.write(str(np.argmax(count_a)) + " " + str(np.argmax(count_b)) + " " + str(np.argmax(count_c)) + " " + str(np.argmax(count_d)) + "\n")
    new_count = [(float)(x) / n for x in count]
    with open(probekeyfile +fgtype+ "_sr.txt", 'a') as f:
        np.savetxt(f, new_count)
    mt=np.mean(time_count)
    with open(probekeyfile + "AW"+fgtype+"_time.txt", 'a') as f:
        f.write(str(mt)+"\n")

'''
KFG: Karatsuba Factor Graph with short cycles
'''
def ComputerBpAwRing(fgtype,noiselevel,filepath,Numberoftripple,data,flag):
    n = 100
    loopnum=5
    probekeyfile=filepath+"sim-template-30-GS"+str(noiselevel)+"-randk/"
    outname = probekeyfile + fgtype+'_attack_result_loop.txt'
    path1_probability = probekeyfile+'data_single.txt'
    path2_probability = probekeyfile+'data_double.txt'
    path3_probability = probekeyfile+'data_four.txt'
    probability1 = np.loadtxt(path1_probability)
    probability2 = np.loadtxt(path2_probability)
    if (flag == 1):
        probability3 = np.loadtxt(path3_probability)
    in_probability1 = probability1.reshape((n, 4, Numberoftripple[0]))
    in_probability2 = probability2.reshape((n, 4, Numberoftripple[1]))
    ck_path = filepath+"ck_index.txt"
    ck_index = np.loadtxt(ck_path, dtype=int)
    count=np.array([0]*5)
    time_count = np.array([0] * n, dtype=float)
    if(flag==1):
        in_probability3 = probability3.reshape((n, Numberoftripple[2]))

    for ff in range(n):
        print(ff)
        begin_time = time()
        mrf = string2factor_graph(
            'f1(a,b,e)f2(a,c,f)f3(b,d,g)f4(c,d,h)f5(a,b,c,d,z)f6(a)f7(b)f8(c)f9(d)f10(e)f11(f)f12(g)f13(h)f14(z)')

        f1 = factor(['a', 'b', 'e'], data[0])
        f2 = factor(['a', 'c', 'f'], data[0])
        f3 = factor(['b', 'd', 'g'], data[0])
        f4 = factor(['c', 'd', 'h'], data[0])
        f5 = factor(['a', 'b', 'c', 'd', 'z'], data[1])

        f6 = factor(['a'], in_probability1[ff, 0])
        f7 = factor(['b'], in_probability1[ff, 1])
        f8 = factor(['c'], in_probability1[ff, 2])
        f9 = factor(['d'], in_probability1[ff, 3])

        f10 = factor(['e'], in_probability2[ff, 0])
        f11 = factor(['f'], in_probability2[ff, 1])
        f12 = factor(['g'], in_probability2[ff, 2])
        f13 = factor(['h'], in_probability2[ff, 3])
        f14 = factor(['z'], in_probability3[ff])

        mrf.change_factor_distribution('f1', f1)
        mrf.change_factor_distribution('f2', f2)
        mrf.change_factor_distribution('f3', f3)
        mrf.change_factor_distribution('f4', f4)
        mrf.change_factor_distribution('f5', f5)

        mrf.change_factor_distribution('f6', f6)
        mrf.change_factor_distribution('f7', f7)
        mrf.change_factor_distribution('f8', f8)
        mrf.change_factor_distribution('f9', f9)

        mrf.change_factor_distribution('f10', f10)
        mrf.change_factor_distribution('f11', f11)
        mrf.change_factor_distribution('f12', f12)
        mrf.change_factor_distribution('f13', f13)
        mrf.change_factor_distribution('f14', f14)

        lbp_a = loopy_belief_propagation(mrf)

        t_a = []
        t_b = []
        t_c = []
        t_d = []
        for i in range(loopnum):
            t_a.append(lbp_a.belief('a', i).get_distribution())
            t_b.append(lbp_a.belief('b', i).get_distribution())
            t_c.append(lbp_a.belief('c', i).get_distribution())
            t_d.append(lbp_a.belief('d', i).get_distribution())

        count_a = t_a[loopnum-1]
        count_b = t_b[loopnum-1]
        count_c = t_c[loopnum-1]
        count_d = t_d[loopnum-1]

        end_time=time()
        time_count[ff]=end_time-begin_time
        if ck_index[ff,0]==np.argmax(count_a):
            count[0]+=1
        if ck_index[ff,1]==np.argmax(count_b):
            count[1]+=1
        if ck_index[ff,3]==np.argmax(count_c):
            count[2]+=1
        if ck_index[ff,4]==np.argmax(count_d):
            count[3]+=1
        if (ck_index[ff,0]==np.argmax(count_a))&(ck_index[ff,1]==np.argmax(count_b))&(ck_index[ff,3]==np.argmax(count_c))&(ck_index[ff,4]==np.argmax(count_d)):
            count[4]+=1
        with open(outname, 'a') as f:
            np.savetxt(f, (count_a, count_b, count_c, count_d))

        with open(probekeyfile+fgtype+"_maxindex_loop.txt", 'a') as f:
            f.write(str(np.argmax(count_a)) + " " + str(np.argmax(count_b)) + " " + str(np.argmax(count_c)) + " " + str(np.argmax(count_d)) + "\n")
    new_count = [(float)(x) / n for x in count]
    with open(probekeyfile +fgtype+ "_sr_loop.txt", 'a') as f:
        np.savetxt(f, new_count)
    mt=np.mean(time_count)
    with open(probekeyfile + "AW"+fgtype+"_time_loop.txt", 'a') as f:
        f.write(str(mt)+"\n")
'''
TFG: Toom-Cook Factor Graph
'''
def ComputerBpWhole(Outputfile,n,loopnum,noiselevel,Numberoftripple,data,tar):
    outname = Outputfile + str(tar)+'_'+str(loopnum)+'_result.txt'
    probability1 = np.loadtxt(file1+"sim-template-30-GS"+str(noiselevel)+"-randk/bayes_attack_result.txt")
    probability2 = np.loadtxt(file2+"sim-template-30-GS"+str(noiselevel)+"-randk/bayes_attack_result.txt")
    probability3 = np.loadtxt(file3+"sim-template-30-GS"+str(noiselevel)+"-randk/bayes_attack_result.txt")
    probability4 = np.loadtxt(file4+"sim-template-30-GS"+str(noiselevel)+"-randk/bayes_attack_result.txt")
    probability5 = np.loadtxt(file5+"sim-template-30-GS"+str(noiselevel)+"-randk/bayes_attack_result.txt")
    probability6 = np.loadtxt(file6+"sim-template-30-GS"+str(noiselevel)+"-randk/bayes_attack_result.txt")
    probability7 = np.loadtxt(file7+"sim-template-30-GS"+str(noiselevel)+"-randk/bayes_attack_result.txt")

    in_probability1 = probability1.reshape(((int)(len(probability1)/4), 4, Numberoftripple[0]))
    in_probability2 = probability2.reshape(((int)(len(probability2)/4), 4, Numberoftripple[1]))
    in_probability3 = probability3.reshape(((int)(len(probability3)/4), 4, Numberoftripple[2]))
    in_probability4 = probability4.reshape(((int)(len(probability4)/4), 4, Numberoftripple[3]))
    in_probability5 = probability5.reshape(((int)(len(probability5)/4), 4, Numberoftripple[4]))
    in_probability6 = probability6.reshape(((int)(len(probability6)/4), 4, Numberoftripple[5]))
    in_probability7 = probability7.reshape(((int)(len(probability7)/4), 4, Numberoftripple[6]))

    for ff in range(n):
        print(ff)

        mrf = string2factor_graph(
            'f1(b3)f2(b3,b2,b1,b0,bw2)f3(b3,b2,b1,b0,bw3)f4(b3,b2,b1,b0,bw4)f5(b3,b2,b1,b0,bw5)f6(b3,b2,b1,b0,bw6)f7(b0)f10(bw2)f11(bw3)f12(bw4)f13(bw5)f14(bw6)')
        f2 = factor(['b3', 'b2', 'b1', 'b0', 'bw2'], data[0])
        f3 = factor(['b3', 'b2', 'b1', 'b0', 'bw3'], data[1])
        f4 = factor(['b3', 'b2', 'b1', 'b0', 'bw4'], data[2])
        f5 = factor(['b3', 'b2', 'b1', 'b0', 'bw5'], data[3])
        f6 = factor(['b3', 'b2', 'b1', 'b0', 'bw6'], data[4])

        f1 = factor(['b3'], in_probability1[ff, tar])
        f7 = factor(['b0'], in_probability7[ff, tar])
        f10 = factor(['bw2'], in_probability2[ff, tar])
        f11 = factor(['bw3'], in_probability3[ff, tar])
        f12 = factor(['bw4'], in_probability4[ff, tar])
        f13 = factor(['bw5'], in_probability5[ff, tar])
        f14 = factor(['bw6'], in_probability6[ff, tar])

        mrf.change_factor_distribution('f1', f1)
        mrf.change_factor_distribution('f2', f2)
        mrf.change_factor_distribution('f3', f3)
        mrf.change_factor_distribution('f4', f4)
        mrf.change_factor_distribution('f5', f5)
        mrf.change_factor_distribution('f6', f6)
        mrf.change_factor_distribution('f7', f7)

        mrf.change_factor_distribution('f10', f10)
        mrf.change_factor_distribution('f11', f11)
        mrf.change_factor_distribution('f12', f12)
        mrf.change_factor_distribution('f13', f13)
        mrf.change_factor_distribution('f14', f14)

        lbp_a = loopy_belief_propagation(mrf)  #run the BP

        t_a = []
        t_b = []
        t_c = []
        t_d = []
        for i in range(loopnum):
            t_a.append(lbp_a.belief('b3', i).get_distribution())
            t_b.append(lbp_a.belief('b2', i).get_distribution())
            t_c.append(lbp_a.belief('b1', i).get_distribution())
            t_d.append(lbp_a.belief('b0', i).get_distribution())


        count_a = t_a[loopnum-1]
        count_b = t_b[loopnum-1]
        count_c = t_c[loopnum-1]
        count_d = t_d[loopnum-1]

        with open(outname, 'a') as f:
            np.savetxt(f, (count_a, count_b, count_c, count_d))
        with open(Outputfile+str(tar)+"_"+str(loopnum)+"_maxindex.txt", 'a') as f:
            f.write(str(np.argmax(count_a)) + " " + str(np.argmax(count_b)) + " " + str(np.argmax(count_c)) + " " + str(np.argmax(count_d)) + "\n")
'''
SFG: Original schoolbook Factor Graph
'''
def FGAwBasic(data,pro,tar,ff):
    loopnum=5
    mrf = string2factor_graph(
        'f1(a,c1)f2(a,c2)f3(a,c3)f4(a,c4)f5(a,c5)f6(a,c6)f7(a,c7)f8(a,c8)f9(a,c9)f10(a,c10)f11(a,c11)f12(a,c12)f13(a,c13)f14(a,c14)f15(a,c15)f16(a,c16)'
        'f18(c1)f19(c2)f20(c3)f21(c4)f22(c5)f23(c6)f24(c7)f25(c8)f26(c9)f27(c10)f28(c11)f29(c12)f30(c13)f31(c14)f32(c15)f33(c16)')
    f1 = factor(['a', 'c1'], data[0])
    f2 = factor(['a', 'c2'], data[1])
    f3 = factor(['a', 'c3'], data[2])
    f4 = factor(['a', 'c4'], data[3])
    f5 = factor(['a', 'c5'], data[4])
    f6 = factor(['a', 'c6'], data[5])
    f7 = factor(['a', 'c7'], data[6])
    f8 = factor(['a', 'c8'], data[7])
    f9 = factor(['a', 'c9'], data[8])
    f10 = factor(['a', 'c10'], data[9])
    f11 = factor(['a', 'c11'], data[10])
    f12 = factor(['a', 'c12'], data[11])
    f13 = factor(['a', 'c13'], data[12])
    f14 = factor(['a', 'c14'], data[13])
    f15 = factor(['a', 'c15'], data[14])
    f16 = factor(['a', 'c16'], data[15])
    f18 = factor(['c1'], pro[ff, 16 * tar + 0])
    f19 = factor(['c2'], pro[ff, 16 * tar + 1])
    f20 = factor(['c3'], pro[ff, 16 * tar + 2])
    f21 = factor(['c4'], pro[ff, 16 * tar + 3])
    f22 = factor(['c5'], pro[ff, 16 * tar + 4])
    f23 = factor(['c6'], pro[ff, 16 * tar + 5])
    f24 = factor(['c7'], pro[ff, 16 * tar + 6])
    f25 = factor(['c8'], pro[ff, 16 * tar + 7])
    f26 = factor(['c9'], pro[ff, 16 * tar + 8])
    f27 = factor(['c10'], pro[ff, 16 * tar + 9])
    f28 = factor(['c11'], pro[ff, 16 * tar + 10])
    f29 = factor(['c12'], pro[ff, 16 * tar + 11])
    f30 = factor(['c13'], pro[ff, 16 * tar + 12])
    f31 = factor(['c14'], pro[ff, 16 * tar + 13])
    f32 = factor(['c15'], pro[ff, 16 * tar + 14])
    f33 = factor(['c16'], pro[ff, 16 * tar + 15])
    mrf.change_factor_distribution('f1', f1)
    mrf.change_factor_distribution('f2', f2)
    mrf.change_factor_distribution('f3', f3)
    mrf.change_factor_distribution('f4', f4)
    mrf.change_factor_distribution('f5', f5)
    mrf.change_factor_distribution('f6', f6)
    mrf.change_factor_distribution('f7', f7)
    mrf.change_factor_distribution('f8', f8)
    mrf.change_factor_distribution('f9', f9)
    mrf.change_factor_distribution('f10', f10)
    mrf.change_factor_distribution('f11', f11)
    mrf.change_factor_distribution('f12', f12)
    mrf.change_factor_distribution('f13', f13)
    mrf.change_factor_distribution('f14', f14)
    mrf.change_factor_distribution('f15', f15)
    mrf.change_factor_distribution('f16', f16)
    mrf.change_factor_distribution('f18', f18)
    mrf.change_factor_distribution('f19', f19)
    mrf.change_factor_distribution('f20', f20)
    mrf.change_factor_distribution('f21', f21)
    mrf.change_factor_distribution('f22', f22)
    mrf.change_factor_distribution('f23', f23)
    mrf.change_factor_distribution('f24', f24)
    mrf.change_factor_distribution('f25', f25)
    mrf.change_factor_distribution('f26', f26)
    mrf.change_factor_distribution('f27', f27)
    mrf.change_factor_distribution('f28', f28)
    mrf.change_factor_distribution('f29', f29)
    mrf.change_factor_distribution('f30', f30)
    mrf.change_factor_distribution('f31', f31)
    mrf.change_factor_distribution('f32', f32)
    mrf.change_factor_distribution('f33', f33)

    lbp_a = loopy_belief_propagation(mrf)
    t_a = []
    for i in range(loopnum):
        t_a.append(lbp_a.belief('a', i).get_distribution())
    count_a = t_a[loopnum-1]
    return count_a

'''
perform the original SFG
'''
def testAwBasic(fgtype,n,noiselevel,filepath,midb_path,Numberoftripple,bw1_one,bw1_two,bw1_four,flag):
    startindex=50000
    prob_path=filepath+ "sim-template-30-GS"+str(noiselevel)+"-randk/probability_attacktrace.txt"
    ck_path=filepath+"ck_index.txt"
    probability = np.loadtxt(prob_path)
    pro = probability.reshape((100, 144, 17))
    outname = filepath+"sim-template-30-GS"+str(noiselevel)+"-randk/"
    all_b = np.loadtxt(midb_path, dtype=int)
    ck_index = np.loadtxt(ck_path, dtype=int)
    attack_b=all_b[16*startindex:16*(startindex+n)]
    b=attack_b.reshape((n,16,9))
    count=np.array([0] * 9)
    time_count = np.array([0] * n,dtype=float)
    new_index=[0,1,3,4,2,5,6,7,8]
    for ff in range(n):
        print(ff)
        begin_time=time()
        middata = gen_single_matrix(Numberoftripple,b, ff,bw1_one,bw1_two,bw1_four,flag)
        in_probability=[]
        for tar in range(9):
            if (tar<8)|(flag==1):
                prob_one=FGAwBasic(middata[16*tar:16*(tar+1)],pro,new_index[tar],ff)
                with open(outname+"Aw"+fgtype+"_prob.txt", 'a') as f:
                    for i in prob_one:
                        f.write(str(i) + " ")
                    f.write("\n")
                assumeindex=np.argmax(prob_one)
                if ck_index[ff,new_index[tar]]==assumeindex:
                    count[tar]+=1
        end_time=time()
        time_count[ff]=end_time-begin_time

    new_count=[(float)(x)/n for x in count]
    with open(outname+ "AW"+fgtype+"_sr.txt", 'a') as f:
        np.savetxt(f, new_count)
    mt=np.mean(time_count)

    with open(outname + "AW"+fgtype+"_time.txt", 'a') as f:
        f.write(str(mt)+"\n")

'''
Bayes' probability
'''
def HWp2keyp(data,pro,tar,ff):
    keyp=np.array([1]*data[0].shape[0],dtype=float)
    for i in range(16):
        for j in range(data[0].shape[0]):
            temp=np.where(data[i][j]==1)
            keyp[j] *= pro[ff,16*tar+i,temp[0]]
    return [t/np.sum(keyp) for t in keyp]

'''
perform Bayes' SFG
'''
def testAwBayes(fgtype,n,noiselevel,filepath,midb_path,Numberoftripple,bw1_one,bw1_two,bw1_four,flag):
    startindex=50000
    prob_path=filepath+ "sim-template-30-GS"+str(noiselevel)+"-randk/probability_attacktrace.txt"
    ck_path=filepath+"ck_index.txt"
    probability = np.loadtxt(prob_path)
    pro = probability.reshape((100, 144, 17))
    outname = filepath+"sim-template-30-GS"+str(noiselevel)+"-randk/"
    all_b = np.loadtxt(midb_path, dtype=int)
    ck_index = np.loadtxt(ck_path, dtype=int)
    attack_b=all_b[16*startindex:16*(startindex+n)]
    b=attack_b.reshape((n,16,9))
    count=np.array([0] * 9)
    time_count = np.array([0] * n,dtype=float)
    new_index=[0,1,3,4,2,5,6,7,8]
    for ff in range(n):
        print(ff)
        begin_time=time()
        middata = gen_single_matrix(Numberoftripple,b, ff,bw1_one,bw1_two,bw1_four,flag)
        in_probability=[]
        for tar in range(9):
            if (tar<8)|(flag==1):
                prob_one=HWp2keyp(middata[16*tar:16*(tar+1)],pro,new_index[tar],ff)
                with open(outname+"Aw"+fgtype+"_prob.txt", 'a') as f:
                    for i in prob_one:
                        f.write(str(i) + " ")
                    f.write("\n")
                assumeindex=np.argmax(prob_one)
                if ck_index[ff,new_index[tar]]==assumeindex:
                    count[tar]+=1
        end_time=time()
        time_count[ff]=end_time-begin_time

    new_count=[(float)(x)/n for x in count]
    with open(outname+ "AW"+fgtype+"_sr.txt", 'a') as f:
        np.savetxt(f, new_count)
    mt=np.mean(time_count)
    with open(outname + "AW"+fgtype+"_time.txt", 'a') as f:
        f.write(str(mt)+"\n")

if __name__ == '__main__':
    nt=[[8,21,65],[624,1464,1],[65,225,1],[65,225,1],[624,1464,1],[624,1464,1],[8,21,65]]  # candidates space
    Numberoftripple=np.array(nt)
    file1="F:/data/saber/1/"
    file2="F:/data/saber/2/"
    file3="F:/data/saber/3/"
    file4="F:/data/saber/4/"
    file5="F:/data/saber/5/"
    file6="F:/data/saber/6/"
    file7="F:/data/saber/7/"
    noiselevel=10    # Gaussian noise optional 2,5,10
    loo_num=50       # Number of traces to attacking test
    bw1_one = [0, 1, 2, 3, 8188, 8189, 8190, 8191]       # candidates of key
    bw1_two = np.loadtxt(file1+"2us.txt", dtype=int)     # candidates
    bw1_four = np.loadtxt(file1+"4us.txt", dtype=int)    # candidates
    data = gen_base_matrix(bw1_one,bw1_two,bw1_four,Numberoftripple[0],1)    # Compute the distribution matrix
    # perform original SFG
    testAwBasic("basic",loo_num,noiselevel,file1,file1+"result/mid-b-aw1.txt",Numberoftripple[0],bw1_one,bw1_two,bw1_four,1)
    # perform bayes' SFG
    testAwBayes("bayes",loo_num,noiselevel,file1,file1+"result/mid-b-aw1.txt",Numberoftripple[0],bw1_one,bw1_two,bw1_four,1)
    slice_probkey("bayes", noiselevel,loo_num, file1,1)
    # perform KFG with no-cycles
    ComputerBpAw("bayes",loo_num,noiselevel,file1,Numberoftripple[0], data,1)
    # perform KFG with short cycles
    ComputerBpAwRing("bayes",noiselevel,file1,Numberoftripple[0], data,1)


    bw2_one = np.loadtxt(file2+"bw2_us.txt", dtype=int)
    bw2_two = np.loadtxt(file2 + "bw2_2us.txt", dtype=int)
    data = gen_base_matrix(bw2_one,bw2_two,bw1_four,Numberoftripple[1],0)
    # perform original SFG
    testAwBasic("basic", loo_num, noiselevel, file2, file2 + "result/mid-b-aw2.txt", Numberoftripple[1], bw2_one,
                bw2_two, bw1_four, 0)
    # perform bayes' SFG
    testAwBayes("bayes",loo_num,noiselevel,file2,file2+"result/mid-b-aw2.txt",Numberoftripple[1],bw2_one,bw2_two,bw1_four,0)

    slice_probkey("bayes", noiselevel,loo_num, file2,0)
    # perform KFG with no-cycles
    ComputerBpAw("bayes",loo_num,noiselevel,file2,Numberoftripple[1], data,0)
    # perform KFG with short cycles
    ComputerBpAwRing("bayes",noiselevel,file2,Numberoftripple[1], data,0)

    bw3_one = np.loadtxt(file3+"bw3_us.txt", dtype=int)
    bw3_two = np.loadtxt(file3 + "bw3_2us.txt", dtype=int)
    data = gen_base_matrix(bw3_one,bw3_two,bw1_four,Numberoftripple[2],0)
    # perform original SFG
    testAwBasic("basic", loo_num, noiselevel, file3, file3 + "result/mid-b-aw3.txt", Numberoftripple[2], bw3_one,
                          bw3_two, bw1_four, 0)
    # perform bayes' SFG
    testAwBayes("bayes", loo_num, noiselevel, file3, file3 + "result/mid-b-aw3.txt", Numberoftripple[2], bw3_one,
                bw3_two, bw1_four, 0)
    slice_probkey("bayes", noiselevel, loo_num, file3, 0)
    # perform KFG with no-cycles
    ComputerBpAw("bayes", loo_num, noiselevel, file3, Numberoftripple[2], data, 0)
    # perform KFG with short cycles
    ComputerBpAwRing("bayes",noiselevel,file3,Numberoftripple[2], data,0)

    bw4_one = np.loadtxt(file4+"bw4_us.txt", dtype=int)
    bw4_two = np.loadtxt(file4 + "bw4_2us.txt", dtype=int)
    data = gen_base_matrix(bw4_one,bw4_two,bw1_four,Numberoftripple[3],0)
    # perform original SFG
    testAwBasic("basic", loo_num, noiselevel, file4, file4 + "result/mid-b-aw4.txt", Numberoftripple[3], bw4_one,
                    bw4_two, bw1_four, 0)
    # perform bayes' SFG
    testAwBayes("bayes", loo_num, noiselevel, file4, file4 + "result/mid-b-aw4.txt", Numberoftripple[3], bw4_one,
                    bw4_two, bw1_four, 0)
    slice_probkey("bayes", noiselevel, loo_num, file4, 0)
    # perform KFG with no-cycles
    ComputerBpAw("bayes", loo_num, noiselevel, file4, Numberoftripple[3], data, 0)
    # perform KFG with short cycles
    ComputerBpAwRing("bayes",noiselevel,file4,Numberoftripple[3], data,0)

    bw5_one = np.loadtxt(file5+"bw5_us.txt", dtype=int)
    bw5_two = np.loadtxt(file5 + "bw5_2us.txt", dtype=int)
    data = gen_base_matrix(bw5_one,bw5_two,bw1_four,Numberoftripple[4],0)
    # perform original SFG
    testAwBasic("basic", loo_num, noiselevel, file5, file5 + "result/mid-b-aw5.txt", Numberoftripple[4], bw5_one,
                                bw5_two, bw1_four, 0)
    # perform bayes' SFG
    testAwBayes("bayes", loo_num, noiselevel, file5, file5 + "result/mid-b-aw5.txt", Numberoftripple[4], bw5_one,
                    bw5_two, bw1_four, 0)
    slice_probkey("bayes", noiselevel, loo_num, file5, 0)
    # perform KFG with no-cycles
    ComputerBpAw("bayes", loo_num, noiselevel, file5, Numberoftripple[4], data, 0)
    # perform KFG with short cycles
    ComputerBpAwRing("bayes",noiselevel,file5,Numberoftripple[4], data,0)

    bw6_one = np.loadtxt(file6+"bw6_us.txt", dtype=int)
    bw6_two = np.loadtxt(file6 + "bw6_2us.txt", dtype=int)
    data = gen_base_matrix(bw6_one,bw6_two,bw1_four,Numberoftripple[5],0)
    # perform original SFG
    testAwBasic("basic", loo_num, noiselevel, file6, file6 + "result/mid-b-aw6.txt", Numberoftripple[5], bw6_one,
                                bw6_two, bw1_four, 0)
    # perform bayes' SFG
    testAwBayes("bayes", loo_num, noiselevel, file6, file6 + "result/mid-b-aw6.txt", Numberoftripple[5], bw6_one,
                    bw6_two, bw1_four, 0)
    slice_probkey("bayes", noiselevel, loo_num, file6, 0)
    # perform KFG with no-cycles
    ComputerBpAw("bayes", loo_num, noiselevel, file6, Numberoftripple[5], data, 0)
    # perform KFG with short cycles
    ComputerBpAwRing("bayes",noiselevel,file6,Numberoftripple[5], data,0)

    bw7_one = [0, 1, 2, 3, 8188, 8189, 8190, 8191]
    bw7_two = np.loadtxt(file7+"2us.txt", dtype=int)
    bw7_four = np.loadtxt(file7+"4us.txt", dtype=int)
    data = gen_base_matrix(bw7_one,bw7_two,bw7_four,Numberoftripple[6],1)
    # perform original SFG
    testAwBasic("basic", loo_num, noiselevel, file7, file7 + "result/mid-b-aw7.txt", Numberoftripple[6], bw7_one,
                   bw7_two, bw1_four, 1)
    # perform bayes' SFG
    testAwBayes("bayes", loo_num, noiselevel, file7, file7 + "result/mid-b-aw7.txt", Numberoftripple[6], bw7_one,
                bw7_two, bw1_four, 1)
    slice_probkey("bayes", noiselevel, loo_num, file7, 1)
    # perform KFG with no-cycles
    ComputerBpAw("bayes", loo_num, noiselevel, file7, Numberoftripple[6], data, 1)
    # perform KFG with short cycles
    ComputerBpAwRing("bayes", noiselevel, file7, Numberoftripple[6], data, 1)

    data=gen_base_whole(bw1_one,bw2_one,bw3_one,bw4_one,bw5_one,bw6_one,Numberoftripple[:,0]) # construct the whole distribution
    for tar in range(4):
        ComputerBpWhole("F:/data/saber/sim/",loo_num,10,noiselevel,Numberoftripple[:,0],data,tar) # perform TFG