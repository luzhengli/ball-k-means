
#include <iostream>
#include <fstream>
#include <time.h>
#include <cstdlib>
#include <algorithm>
#include "Eigen/Dense"
#include <vector>
#include <cfloat>

using namespace std;
using namespace Eigen;
std::ofstream outFile;

typedef float OurType;

typedef VectorXf VectorOur;

typedef MatrixXf MatrixOur;

typedef vector<vector<OurType>> ClusterDistVector;

typedef vector<vector<unsigned int>> ClusterIndexVector;

typedef Array<bool, 1, Dynamic> VectorXb;

typedef struct Neighbor
//Define the "neighbor" structure
{
    OurType distance;
    int index;
};

typedef vector<Neighbor> sortedNeighbors;

MatrixOur load_data(const char *filename);

inline MatrixOur
update_centroids(MatrixOur &dataset, ClusterIndexVector &cluster_point_index, unsigned int k, unsigned int n,
                 VectorXb &flag,
                 unsigned int iteration_counter, MatrixOur &old_centroids);

inline void update_radius(MatrixOur &dataset, ClusterIndexVector &cluster_point_index, MatrixOur &new_centroids,
                          ClusterDistVector &temp_dis,
                          VectorOur &the_rs, VectorXb &flag, unsigned int iteration_counter, unsigned int &cal_dist_num,
                          unsigned int the_rs_size);

inline sortedNeighbors
get_sorted_neighbors_Ring(VectorOur &the_Rs, MatrixOur &centers_dis, unsigned int now_ball, unsigned int k,
                          vector<unsigned int> &now_center_index);

inline sortedNeighbors
get_sorted_neighbors_noRing(VectorOur &the_rs, MatrixOur &centers_dist, unsigned int now_ball, unsigned int k,
                            vector<unsigned int> &now_center_index);

inline void
cal_centers_dist(MatrixOur &new_centroids, unsigned int iteration_counter, unsigned int k, VectorOur &the_rs,
                 VectorOur &delta, MatrixOur &centers_dis);

inline MatrixOur cal_dist(MatrixOur &dataset, MatrixOur &centroids);

inline MatrixOur
cal_ring_dist_Ring(unsigned j, unsigned int data_num, unsigned int dataset_cols, MatrixOur &dataset,
                   MatrixOur &now_centers,
                   ClusterIndexVector &now_data_index);

inline MatrixOur
cal_ring_dist_noRing(unsigned int data_num, unsigned int dataset_cols, MatrixOur &dataset, MatrixOur &now_centers,
                     vector<unsigned int> &now_data_index);

void initialize(MatrixOur &dataset, MatrixOur &centroids, VectorXi &labels, ClusterIndexVector &cluster_point_index,
                ClusterIndexVector &clusters_neighbors_index,
                ClusterDistVector &temp_dis);

VectorXi ball_k_means_Ring(MatrixOur &dataset, MatrixOur &centroids, bool detail = false)
{

    double start_time, end_time;

    bool judge = true;

    const unsigned int dataset_rows = dataset.rows();
    const unsigned int dataset_cols = dataset.cols();
    const unsigned int k = centroids.rows();

    ClusterIndexVector temp_cluster_point_index;
    ClusterIndexVector cluster_point_index;
    ClusterIndexVector clusters_neighbors_index;
    //二维向量 每个一维元素表示一层环域的所有点的索引 如：now_data_index[j-1]表示第j层(j>=1)环域
    ClusterIndexVector now_data_index;
    //二维向量 每个行元素表示一个簇中点到质心的距离
    ClusterDistVector temp_dis;

    MatrixOur new_centroids(k, dataset_cols);
    MatrixOur old_centroids = centroids;
    MatrixOur centers_dis(k, k);

    VectorXb flag(k);
    VectorXb old_flag(k);

    VectorXi labels(dataset_rows);
    VectorOur delta(k);

    vector<unsigned int> old_now_index;
    vector<OurType> distance_arr;

    VectorOur the_rs(k);
    // 当前簇的近邻簇的个数
    unsigned int now_centers_rows;
    unsigned int iteration_counter;
    unsigned int num_of_neighbour;
    unsigned int neighbour_num;
    unsigned int cal_dist_num;
    unsigned int data_num;

    MatrixOur::Index minCol;
    new_centroids.setZero();
    iteration_counter = 0;
    num_of_neighbour = 0;
    cal_dist_num = 0;
    flag.setZero();

    //initialize cluster_point_index and temp_dis
    initialize(dataset, centroids, labels, cluster_point_index, clusters_neighbors_index, temp_dis);

    temp_cluster_point_index.assign(cluster_point_index.begin(), cluster_point_index.end());

    start_time = clock();

    while (true)
    {
        old_flag = flag;
        //record cluster_point_index from the previous round
        cluster_point_index.assign(temp_cluster_point_index.begin(), temp_cluster_point_index.end());
        iteration_counter += 1;

        //update the matrix of centroids
        new_centroids = update_centroids(dataset, cluster_point_index, k, dataset_cols, flag, iteration_counter,
                                         old_centroids);

        if (new_centroids != old_centroids)
        {
            //delta: distance between each center and the previous center
            delta = (((new_centroids - old_centroids).rowwise().squaredNorm())).array().sqrt();

            old_centroids = new_centroids;

            //get the radius of each centroids
            update_radius(dataset, cluster_point_index, new_centroids, temp_dis, the_rs, flag, iteration_counter,
                          cal_dist_num,
                          k);
            //Calculate distance between centers

            cal_centers_dist(new_centroids, iteration_counter, k, the_rs, delta, centers_dis);

            flag.setZero();

            //returns the set of neighbors

            //nowball;
            unsigned int now_num = 0;
            for (unsigned int now_ball = 0; now_ball < k; now_ball++)
            {
                //返回当前簇的近邻簇（包括当前簇 按照距离升序返回）
                sortedNeighbors neighbors = get_sorted_neighbors_Ring(the_rs, centers_dis, now_ball, k,
                                                                      clusters_neighbors_index[now_ball]);
                //当前簇中的点个数
                now_num = temp_dis[now_ball].size();
                if (the_rs(now_ball) == 0)
                    continue;

                //Get the coordinates of the neighbors and neighbors of the current ball
                old_now_index.clear();
                old_now_index.assign(clusters_neighbors_index[now_ball].begin(),
                                     clusters_neighbors_index[now_ball].end());
                clusters_neighbors_index[now_ball].clear();
                neighbour_num = neighbors.size();
                MatrixOur now_centers(neighbour_num, dataset_cols);

                for (unsigned int i = 0; i < neighbour_num; i++)
                {
                    clusters_neighbors_index[now_ball].push_back(neighbors[i].index);
                    now_centers.row(i) = new_centroids.row(neighbors[i].index);
                }

                num_of_neighbour += neighbour_num;

                now_centers_rows = now_centers.rows();

                judge = true;

                if (clusters_neighbors_index[now_ball] != old_now_index)
                    judge = false;
                else
                {
                    for (int i = 0; i < clusters_neighbors_index[now_ball].size(); i++)
                    {
                        if (old_flag(clusters_neighbors_index[now_ball][i]) != false)
                        {
                            judge = false;
                            break;
                        }
                    }
                }

                if (judge)
                {
                    continue;
                }

                now_data_index.clear();
                distance_arr.clear();

                for (unsigned int j = 1; j < neighbour_num; j++)
                {
                    //将当前簇和其近邻簇的距离的一半记录下来 【疑问】这里可以用`neighbors[j].distance`去代替`centers_dis(clusters_neighbors_index[now_ball][j], now_ball)`吧？ 没必要重新计算一次距离
                    distance_arr.push_back(centers_dis(clusters_neighbors_index[now_ball][j], now_ball) / 2);
                    now_data_index.push_back(vector<unsigned int>());
                }

                // （重要）将当前簇的点划分到当前簇的第i环域中
                // 遍历当前簇的每个点
                for (unsigned int i = 0; i < now_num; i++)
                {
                    // 遍历每个近邻簇
                    for (unsigned int j = 1; j < neighbour_num; j++)
                    {
                        // 比较 当前点到当前簇的距离 与 当前点与近邻簇的距离 ， 将当前点分配到当前簇的第j环域中 有关第j环域的定义见论文笔记的`定义4`
                        // 如果当前簇是最远的近邻簇 且 当前点到当前簇（所属簇）质心的距离>当前簇到最远近邻簇距离的一半
                        if (j == now_centers_rows - 1 && temp_dis[now_ball][i] > distance_arr[j - 1])
                        {
                            // 将该点分配到最外层环域（点的索引记录到最外层环域）
                            now_data_index[j - 1].push_back(cluster_point_index[now_ball][i]);
                            break;
                        }
                        // 如果当前簇（记为第j个近邻簇）不是最远的近邻簇 且
                        // 当前簇到第j近邻簇的距离的一半 < 当前点到当前簇（所属簇）的距离 <= 当前簇到第j+1近邻簇的距离的一半
                        if (j != now_centers_rows - 1 && temp_dis[now_ball][i] > distance_arr[j - 1] &&
                            temp_dis[now_ball][i] <= distance_arr[j])
                        {
                            // 将该点分配到第j层环域（点的索引记录到第j层环域）
                            now_data_index[j - 1].push_back(cluster_point_index[now_ball][i]);
                            break;
                        }
                    }
                }

                judge = false;
                int lenth = old_now_index.size();

                // 遍历各层环域 对其中的点进行再分配
                // ？得知了环域划分情况，点如何进行移动呢？
                for (unsigned int j = 1; j < neighbour_num; j++)
                {
                    // 第j环域的点个数
                    data_num = now_data_index[j - 1].size();
                    // 如果第j环域点个数为0 则该层环域不做处理
                    if (data_num == 0)
                        continue;

                    // 距离
                    MatrixOur temp_distance = cal_ring_dist_Ring(j, data_num, dataset_cols, dataset, now_centers,
                                                                 now_data_index);

                    // 距离计算次数
                    cal_dist_num += data_num * j;

                    // 距离当前点最近的近邻簇（包含当前簇）在数据集中的索引
                    unsigned int new_label;
                    // 遍历第j层环域的所有点
                    for (unsigned int i = 0; i < data_num; i++)
                    {
                        // 记录距离当前点最近的近邻簇在距离矩阵中的位置
                        temp_distance.row(i).minCoeff(&minCol);
                        // 记录距离当前点最近的近邻簇（包含当前簇）在数据集中的索引（即当前点的新的所属簇的索引）
                        new_label = clusters_neighbors_index[now_ball][minCol];
                        //如果当前点所属的簇不是距离该点最近的近邻簇（包括当前簇） 则进行以下操作：
                        //1.标记当前簇 和 当前点新的所属簇 发生了变化 flag(now_ball) = true 、 flag(new_label)=true
                        //2.把当前环域的点从原来的簇中点删除 然后将其放到最近邻簇中 并更新该点所属的簇索引
                        if (labels[now_data_index[j - 1][i]] != new_label)
                        {
                            flag(now_ball) = true;
                            flag(new_label) = true;

                            //Update localand global labels
                            vector<unsigned int>::iterator it = (temp_cluster_point_index[labels[now_data_index[j - 1][i]]]).begin();
                            while ((it) != (temp_cluster_point_index[labels[now_data_index[j - 1][i]]]).end())
                            {
                                if (*it == now_data_index[j - 1][i])
                                {
                                    it = (temp_cluster_point_index[labels[now_data_index[j - 1][i]]]).erase(it);
                                    break;
                                }
                                else
                                    ++it;
                            }
                            temp_cluster_point_index[new_label].push_back(now_data_index[j - 1][i]);
                            labels[now_data_index[j - 1][i]] = new_label;
                        }
                    }
                }
            }
        }
        else
            break;
    }
    end_time = clock();

    if (detail == true)
    {
        cout << "ball-k-means with dividing ring:" << endl;
        cout << "k                :                  ||" << k << endl;
        cout << "iterations       :                  ||" << iteration_counter << endl;
        cout << "The number of calculating distance: ||" << cal_dist_num << endl;
        cout << "The number of neighbors:            ||" << num_of_neighbour << endl;
        cout << "Time per round:                     ||"
             << (double)(end_time - start_time) / CLOCKS_PER_SEC * 1000 / iteration_counter << endl;
    }
    return labels;
}

VectorXi ball_k_means_noRing(MatrixOur &dataset, MatrixOur &centroids, bool detail = false)
{

    double start_time, end_time;

    bool judge = true;

    //数据集行数
    const unsigned int dataset_rows = dataset.rows();
    //数据集列数
    const unsigned int dataset_cols = dataset.cols();
    //聚类数
    const unsigned int k = centroids.rows();
    //稳定域的半径
    OurType stable_field_dist = 0;
    //？
    ClusterIndexVector temp_clusters_point_index;
    //一个二维向量 每个一维向量元素表示一个簇内的所有样本的索引
    ClusterIndexVector clusters_point_index;
    //这一轮迭代中当前簇的所有近邻簇的索引（包含自身 且第一个就是当前簇的索引） 一个二维向量 每个一维向量元素表示一个簇的所有近邻簇的索引
    ClusterIndexVector clusters_neighbors_index;
    //一个二维向量 每个一维向量元素表示一个簇中所有点到簇中心的距离
    ClusterDistVector point_center_dist;

    //本轮迭代的聚类中心坐标 二维矩阵 每行表示一个聚类中心的坐标
    MatrixOur new_centroids(k, dataset_cols);
    //上一轮迭代的聚类中心坐标
    MatrixOur old_centroids = centroids;
    //聚类中心的距离
    MatrixOur centers_dist(k, k);
    //本轮迭代球簇的标志向量（簇是否变化） 标志为0表示不变化 为1表示变化
    VectorXb flag(k);
    //上一轮迭代球簇的标志量（簇是否变化） 标志为0表示不变化 为1表示变化
    VectorXb old_flag(k);
    //数据集样本所属的簇标签
    VectorXi labels(dataset_rows);
    //前一轮质心与后一轮质心的距离差（质心偏移量：根据第n-1迭代可以推知第n次迭代是否具有近邻关系 delta就是第n次迭代中要计算的一个量）
    VectorOur delta(k);
    //一维向量 当前簇中处于环域的所有点的索引
    vector<unsigned int> now_data_index;
    //上一轮迭代中簇的所有近邻簇的索引 一维向量
    vector<unsigned int> old_now_index;
    //所有球簇的半径 一维向量
    VectorOur the_rs(k);
    //这一轮迭代中 当前簇的所有近邻簇的总数（包含当前簇）
    unsigned int now_centers_rows;
    //迭代次数
    unsigned int iteration_counter;
    //近邻的总数（包括自身）
    unsigned int num_of_neighbour;
    //当前球簇的近邻数（包括自身）
    unsigned int neighbour_num;
    //计算距离的次数
    unsigned int cal_dist_num;
    //当前簇中未处于稳定域内的点的总数
    unsigned int data_num;

    //bool key = true;
    //参数的初始化
    //？
    MatrixOur::Index minCol;
    //？
    new_centroids.setZero();
    iteration_counter = 0;
    num_of_neighbour = 0;
    cal_dist_num = 0;
    //开始时：设置所有簇都不变化（flag=0）
    flag.setZero();

    //1.将样本点分配到初始点为质心的簇中 initialize clusters_point_index and point_center_dist
    initialize(dataset, centroids, labels, clusters_point_index, clusters_neighbors_index, point_center_dist);

    //把clusters_point_index的内容拷贝给temp_clusters_point_index
    temp_clusters_point_index.assign(clusters_point_index.begin(), clusters_point_index.end());

    start_time = clock();
    //2.迭代更新质心过程
    while (true)
    {
        //保存上一轮迭代的球簇标志向量
        old_flag = flag;
        //record clusters_point_index from the previous round
        clusters_point_index.assign(temp_clusters_point_index.begin(), temp_clusters_point_index.end());
        //迭代次数+1
        iteration_counter += 1;

        // //【测试】保存更新前的质心坐标
        // std::ofstream outFile;
        // outFile.open("/home/ubuntu/workspace/python/luyao/experiment/ball-k-means-master/C++Version/old_centroids.txt");
        // outFile << old_centroids;
        // outFile.close();

        //通过各个簇的点更新计算得到各个簇的质心坐标 如果 簇不稳定或迭代次数为1：更新质心坐标； 否则（簇稳定、迭代次数不为0）：质心坐标不变
        new_centroids = update_centroids(dataset, clusters_point_index, k, dataset_cols, flag, iteration_counter,
                                         old_centroids);

        // //【测试】保存更新后的质心坐标
        // outFile.open("/home/ubuntu/workspace/python/luyao/experiment/ball-k-means-master/C++Version/new_centroids.txt");
        // outFile << new_centroids;
        // outFile.close();

        //如果存在簇的质心坐标发生改变 则进行以下操作 否则迭代过程结束 算法退出
        //主要操作如下：
        //1.计算本轮迭代质心坐标与上一轮质心坐标的位移差delta
        //2.获取各个球簇的半径（球簇半径=各个球簇中点到其质心的最远距离）
        //3.计算质心间的两两距离centers_dist
        if (new_centroids != old_centroids) // 判断两个二维矩阵是否相等
        {
            //delta: distance between each center and the previous center
            delta = (((new_centroids - old_centroids).rowwise().squaredNorm())).array().sqrt(); //`.rowwise().squaredNorm()`表示每个行向量的内积（行向量对应元素相乘然后加和）

            // //【测试】保存delta
            // outFile.open("/home/ubuntu/workspace/python/luyao/experiment/ball-k-means-master/C++Version/delta.txt");
            // outFile << delta;
            // outFile.close();

            //保存本轮迭代中的质心矩阵
            old_centroids = new_centroids;
            //获取每个球簇的半径 get the radius of each centroids（更新变量：point_center_dist、the_rs、cal_dist_num）
            update_radius(dataset, clusters_point_index, new_centroids, point_center_dist, the_rs, flag,
                          iteration_counter, cal_dist_num, k);

            //计算质心间两两距离 Calculate distance between centers
            cal_centers_dist(new_centroids, iteration_counter, k, the_rs, delta, centers_dist);

            // // 【测试】保存flag
            // outFile.open("/home/ubuntu/workspace/python/luyao/experiment/ball-k-means-master/C++Version/flag_before.txt");
            // outFile << flag;
            // outFile.close();
            flag.setZero();

            // // 【测试】保存flag
            // outFile.open("/home/ubuntu/workspace/python/luyao/experiment/ball-k-means-master/C++Version/flag_after.txt");
            // outFile << flag;
            // outFile.close();

            //returns the set of neighbors
            //当前簇内点的总数
            unsigned int now_num = 0;
            for (unsigned int now_ball = 0; now_ball < k; now_ball++)
            {
                //获取当前簇的近邻簇的距离和索引信息（包含自身 按索引大小排序 从0到n）
                sortedNeighbors neighbors = get_sorted_neighbors_noRing(the_rs, centers_dist, now_ball, k,
                                                                        clusters_neighbors_index[now_ball]);
                //获取当前簇内的点数
                now_num = point_center_dist[now_ball].size();
                //如果当前簇半径为0 则本次迭代结束
                if (the_rs(now_ball) == 0)
                    continue;

                //保存上一轮迭代中当前簇的所有近邻簇的索引 Get the coordinates of the neighbors and neighbors of the current ball
                old_now_index.clear();
                old_now_index.assign(clusters_neighbors_index[now_ball].begin(),
                                     clusters_neighbors_index[now_ball].end());
                //清空这一轮簇的所有近邻簇的索引 为后面更新做准备
                clusters_neighbors_index[now_ball].clear();
                //获取当前簇的近邻簇数（包括自身）
                neighbour_num = neighbors.size();
                //这一轮迭代中簇的所有近邻簇质心的坐标（包含自身） 二维矩阵 每个行元素表示一个簇质心的坐标
                MatrixOur now_centers(neighbour_num, dataset_cols);

                //获取当前簇的所有近邻簇的索引（包含自身 且第一个就是当前簇的索引） 然后根据获取当前簇的所有近邻簇质心的坐标
                for (unsigned int i = 0; i < neighbour_num; i++)
                {
                    //参数`clusters_neighbors_index`是在这里更新的！
                    clusters_neighbors_index[now_ball].push_back(neighbors[i].index);
                    //记录本轮迭代中 当前簇的所有近邻簇质心的坐标（包含自身 且第一个就是当前簇的坐标）
                    now_centers.row(i) = new_centroids.row(neighbors[i].index);
                }

                //每个簇的所有近邻簇之和
                num_of_neighbour += neighbour_num;

                now_centers_rows = now_centers.rows();

                //判断当前簇与近邻簇间的点是否需要移动 judge=true表示不需移动 judge=false表示需要移动
                //簇间点是否移动的标志量
                judge = true;
                //如果当前簇的近邻关系发生改变 则设置judge为false 表示需要移动点
                if (clusters_neighbors_index[now_ball] != old_now_index)
                    judge = false;
                //如果当前簇的近邻关系没发生改变 但是其近邻簇发生了改变 也需设置judge为false 不是需要移动点
                else
                {
                    for (int i = 0; i < clusters_neighbors_index[now_ball].size(); i++)
                    {
                        if (old_flag(clusters_neighbors_index[now_ball][i]) != false)
                        {
                            judge = false;
                            break;
                        }
                    }
                }
                //簇间点不需移动-无需在当前簇和其近邻间移动点
                if (judge)
                {
                    continue;
                }

                //开始移动当前簇与近邻簇间的点
                //清空 当前簇中所有未处于稳定域的点索引
                now_data_index.clear();
                //初始化变量保存 稳定域的半径：开始将稳定域的半径设置为当前簇的半径
                stable_field_dist = the_rs(now_ball);
                //通过迭代当前簇的近邻簇 求当前簇的稳定域的半径
                for (unsigned int j = 1; j < neighbour_num; j++)
                {
                    stable_field_dist = min(stable_field_dist,
                                            centers_dist(clusters_neighbors_index[now_ball][j], now_ball) / 2);
                }
                //比较 当前簇中点到质心的距离 与 该簇稳定域的半径 的大小，将处于环域的点记录下来（保存索引）
                for (unsigned int i = 0; i < now_num; i++)
                {

                    if (point_center_dist[now_ball][i] > stable_field_dist)
                    {
                        now_data_index.push_back(clusters_point_index[now_ball][i]);
                    }
                }
                //当前簇中未处于稳定域的点的总数
                data_num = now_data_index.size();
                //如果当前簇中所有点都在稳定域中（当前簇稳定） 则结束当前簇的迭代
                if (data_num == 0)
                {
                    continue;
                }
                //二维矩阵 处于当前簇环域（即不位于稳定域）中的所有点到其近邻簇（包含当前簇）的距离矩阵
                MatrixOur temp_distance = cal_ring_dist_noRing(data_num, dataset_cols, dataset, now_centers,
                                                               now_data_index);
                //统计一个簇中未处于稳定域的点与近邻簇（自身也考虑）之间的距离计算总次数
                cal_dist_num += data_num * now_centers.rows();

                //距离当前点最近的近邻簇（包含当前簇）在数据集中的索引
                unsigned int new_label;
                //遍历环域中所有的点
                for (unsigned int i = 0; i < data_num; i++)
                {
                    //记录距离当前点最近的近邻簇在距离矩阵中的位置
                    temp_distance.row(i).minCoeff(&minCol); //clusters_neighbors_index(minCol)
                    //记录距离当前点最近的近邻簇（包含当前簇）在数据集中的索引（即当前点的新的所属簇的索引）
                    new_label = clusters_neighbors_index[now_ball][minCol];
                    //如果当前点所属的簇不是距离该点最近的近邻簇（包括当前簇） 则进行以下操作：
                    //1.标记当前簇 和 当前点新的所属簇 发生了变化 flag(now_ball) = true 、 flag(new_label)=true
                    //2.把当前环域的点从原来的簇中点删除 然后将其放到最近邻簇中 并更新该点所属的簇索引
                    if (labels[now_data_index[i]] != new_label)
                    {

                        flag(now_ball) = true;
                        flag(new_label) = true;

                        //Update localand global labels
                        //把当前环域的点从原来的簇中点删除 然后将其放到最近邻簇中
                        vector<unsigned int>::iterator it = (temp_clusters_point_index[labels[now_data_index[i]]]).begin();
                        while ((it) != (temp_clusters_point_index[labels[now_data_index[i]]]).end())
                        {
                            if (*it == now_data_index[i])
                            {
                                it = (temp_clusters_point_index[labels[now_data_index[i]]]).erase(it);
                                break;
                            }
                            else
                            {
                                ++it;
                            }
                        }
                        temp_clusters_point_index[new_label].push_back(now_data_index[i]);
                        labels[now_data_index[i]] = new_label;
                    }
                }
            }
        }
        //所有质心状态都已稳定（质心坐标都不变） 算法退出
        else
        {
            break;
        }
    }
    end_time = clock();

    //输出一些中间计算信息
    if (detail == true)
    {
        cout << "ball-k-means without dividing ring:" << endl;
        cout << "k                :                  ||" << k << endl;
        cout << "iterations       :                  ||" << iteration_counter << endl;
        cout << "The number of calculating distance: ||" << cal_dist_num << endl;
        cout << "The number of neighbors:            ||" << num_of_neighbour << endl;
        cout << "Time per round:                     ||" << (double)(end_time - start_time) / CLOCKS_PER_SEC * 1000 / iteration_counter << endl;
    }
    return labels;
}

MatrixOur load_data(const char *filename)
{
    /*

    *Summary: Read data through file path

    *Parameters:

    *     filename: file path.*    

    *Return : Dataset in eigen matrix format.

    */

    int x = 0, y = 0; // x: rows  ，  y/x: cols
    ifstream inFile(filename, ios::in);
    string lineStr;
    while (getline(inFile, lineStr))
    {
        stringstream ss(lineStr);
        string str;
        while (getline(ss, str, ','))
            y++;
        x++;
    }
    MatrixOur data(x, y / x);
    ifstream inFile2(filename, ios::in);
    string lineStr2;
    int i = 0;
    while (getline(inFile2, lineStr2))
    {
        stringstream ss2(lineStr2);
        string str2;
        int j = 0;
        while (getline(ss2, str2, ','))
        {
            data(i, j) = atof(const_cast<const char *>(str2.c_str()));
            j++;
        }
        i++;
    }
    return data;
}

inline MatrixOur
update_centroids(MatrixOur &dataset, ClusterIndexVector &cluster_point_index, unsigned int k, unsigned int n,
                 VectorXb &flag, unsigned int iteration_counter, MatrixOur &old_centroids)
{
    /*

    *Summary: Update the center point of each cluster
    如果 簇不稳定 或 迭代次数等于1：更新簇的质心坐标
    如果 簇稳定 或 迭代次数不为1：簇的质心坐标不变

    *Parameters:

    *     dataset: dataset in eigen matrix format.*   

    *     clusters_point_index: global position of each point in the cluster.* 

    *     k: number of center points.*  

    *     dataset_cols: data set dimensions*  

    *     flag: judgment label for whether each cluster has changed.*  

    *     iteration_counter: number of iterations.*  

    *     old_centroids: center matrix of previous round.*  

    *Return : updated center matrix.

    */
    //簇中点的数量
    unsigned int cluster_point_index_size = 0;
    unsigned int temp_num = 0;
    //更新后的质心坐标 二维 每一行表示一个质心的坐标
    MatrixOur new_c(k, n);
    //中间变量 表示各个簇中所有点坐标的和
    VectorOur temp_array(n);
    for (unsigned int i = 0; i < k; i++)
    {
        //中间变量 记录一个簇中所有点的个数
        temp_num = 0;
        temp_array.setZero();
        cluster_point_index_size = cluster_point_index[i].size();
        //根据各个簇的点 更新簇的中心（更新后簇质心=上一轮中簇所有点的重心）
        //如果 簇发生变化 或 迭代次数等于1：更新簇的质心坐标
        if (flag(i) != 0 || iteration_counter == 1)
        {
            for (unsigned int j = 0; j < cluster_point_index_size; j++)
            {
                temp_array += dataset.row(cluster_point_index[i][j]);
                temp_num++;
            }
            //更新后簇的质心=0 or 簇中所有点的重心
            new_c.row(i) = (temp_num != 0) ? (temp_array / temp_num) : temp_array;
        }
        //如果 簇稳定 或 迭代次数不为1：簇的质心坐标不变
        else
            new_c.row(i) = old_centroids.row(i);
    }
    return new_c;
}

inline void update_radius(MatrixOur &dataset, ClusterIndexVector &cluster_point_index, MatrixOur &new_centroids,
                          ClusterDistVector &temp_dis, VectorOur &the_rs, VectorXb &flag,
                          unsigned int iteration_counter,
                          unsigned int &cal_dist_num, unsigned int the_rs_size)
{

    /*

    *Summary: Update the radius of each cluster

    *Parameters:

    *     dataset: dataset in eigen matrix format.*   

    *     clusters_point_index: global position of each point in the cluster.* 

    *     new_centroids: updated center matrix.*  

    *     point_center_dist/temp_dis: distance from point in cluster to center*  

    *     the_rs: The radius of each cluster.*  

    *     flag: judgment label for whether each cluster has changed.*  

    *     iteration_counter: number of iterations.*  

    *     cal_dist_num: distance calculation times.* 

    *     the_rs_size: number of clusters.* 

    */

    // //【测试】保存flag向量
    // std::ofstream outFile;
    // outFile.open("/home/ubuntu/workspace/python/luyao/experiment/ball-k-means-master/C++Version/flag.txt");
    // outFile << flag;
    // outFile.close();

    OurType temp = 0;
    unsigned int cluster_point_index_size = 0;
    //如果簇发生变化（不稳定） 或 迭代次数为1：进行以下操作：
    //1.计算簇中所有点距离质心的距离 保存到变量temp_dis
    //2.计算簇的半径（簇的半径=max{dictance(x_i, centroid), i∈该簇中的点数}）
    for (unsigned int i = 0; i < the_rs_size; i++)
    {
        cluster_point_index_size = cluster_point_index[i].size();
        if (flag(i) != 0 || iteration_counter == 1)
        {
            the_rs(i) = 0;
            temp_dis[i].clear();
            //对于当前簇中每个点：进行以下操作
            for (unsigned int j = 0; j < cluster_point_index_size; j++)
            {
                //距离计算次数加1
                cal_dist_num++;
                //计算点与所在簇质心间的距离
                temp = sqrt((new_centroids.row(i) - dataset.row(cluster_point_index[i][j])).squaredNorm());
                temp_dis[i].push_back(temp);
                //将簇中距离质心最远的点的距离作为该簇的半径
                if (the_rs(i) < temp)
                    the_rs(i) = temp;
            }
        }
    }
};

bool LessSort(Neighbor a, Neighbor b)
{
    return (a.distance < b.distance);
}

//todo
inline sortedNeighbors
get_sorted_neighbors_Ring(VectorOur &the_Rs, MatrixOur &centers_dis, unsigned int now_ball, unsigned int k,
                          vector<unsigned int> &now_center_index)
{

    /*

    *Summary: Get the sorted neighbors

    *Parameters:

    *     the_rs: the radius of each cluster.*   

    *     centers_dist: distance matrix between centers.* 

    *     now_ball: current ball label.*  

    *     k: number of center points*  

    *     now_center_index: nearest neighbor label of the current ball.*  

    */

    VectorXi flag = VectorXi::Zero(k);
    sortedNeighbors neighbors;

    Neighbor temp;
    temp.distance = 0;
    temp.index = now_ball;
    neighbors.push_back(temp);
    flag(now_ball) = 1;

    for (unsigned int j = 1; j < now_center_index.size(); j++)
    {
        if (centers_dis(now_ball, now_center_index[j]) == 0 ||
            2 * the_Rs(now_ball) - centers_dis(now_ball, now_center_index[j]) < 0)
        {
            flag(now_center_index[j]) = 1;
        }
        else
        {
            flag(now_center_index[j]) = 1;
            temp.distance = centers_dis(now_ball, now_center_index[j]);
            temp.index = now_center_index[j];
            neighbors.push_back(temp);
        }
    }

    for (unsigned int j = 0; j < k; j++)
    {
        if (flag(j) == 1)
        {
            continue;
        }
        if (centers_dis(now_ball, j) != 0 && 2 * the_Rs(now_ball) - centers_dis(now_ball, j) >= 0)
        {
            temp.distance = centers_dis(now_ball, j);
            temp.index = j;
            neighbors.push_back(temp);
        }
    }

    sort(neighbors.begin(), neighbors.end(), LessSort);
    return neighbors;
}

inline sortedNeighbors
get_sorted_neighbors_noRing(VectorOur &the_rs, MatrixOur &centers_dist, unsigned int now_ball, unsigned int k,
                            vector<unsigned int> &now_center_index)
{
    /*

    *Summary: 获取一个簇的所有近邻 返回结果按照簇的索引大小排序（从0-n）

    *Parameters:

    *     the_rs: the radius of each cluster.*   

    *     centers_dist: distance matrix between centers.* 

    *     now_ball: current ball label.*  

    *     k: number of center points*  

    *     now_center_index: nearest neighbor label of the current ball.*  

    */
    //一维向量 标记簇是否访问过 flag=1表示已经访问过
    VectorXi flag = VectorXi::Zero(k);
    //每个近邻由一个结构类型表示 包含距离和索引两部分
    //一个簇可能有多个近邻 需要一个保存近邻结构类型的向量
    sortedNeighbors neighbors;

    Neighbor temp;
    temp.distance = 0;
    temp.index = now_ball;
    neighbors.push_back(temp);
    flag(now_ball) = 1;
    //找到当前簇的所有近邻 将近邻信息保存到neighbors

    //
    for (unsigned int j = 1; j < now_center_index.size(); j++)
    {
        //如果 两个簇质心距离为0 or 不满足近邻定义 则标记已经判断过
        if (centers_dist(now_ball, now_center_index[j]) == 0 ||
            2 * the_rs(now_ball) - centers_dist(now_ball, now_center_index[j]) < 0) //？这里`centers_dist(now_ball, now_center_index[j]) < 0`应该改成≤吧
        {
            flag(now_center_index[j]) = 1;
        }
        //否则 记录近邻簇的距离和索引
        else
        {
            flag(now_center_index[j]) = 1;
            temp.distance = centers_dist(now_ball, now_center_index[j]);
            temp.index = now_center_index[j];
            neighbors.push_back(temp);
        }
    }

    //找到当前簇的所有近邻 将近邻信息保存到neighbors
    for (unsigned int j = 0; j < k; j++)
    {
        if (flag(j) == 1)
        {
            continue;
        }
        if (centers_dist(now_ball, j) != 0 && 2 * the_rs(now_ball) - centers_dist(now_ball, j) >= 0)
        {
            temp.distance = centers_dist(now_ball, j);
            temp.index = j;
            neighbors.push_back(temp);
        }
    }

    return neighbors;
}

inline void
cal_centers_dist(MatrixOur &new_centroids, unsigned int iteration_counter, unsigned int k, VectorOur &the_rs,
                 VectorOur &delta, MatrixOur &centers_dis)
{

    /*

    *Summary: Calculate the distance matrix between center points

    *Parameters:

    *     new_centroids: current center matrix.*   

    *     iteration_counter: number of iterations.* 

    *     k: number of center points.*  

    *     the_rs: the radius of each cluster*  

    *     delta: distance between each center and the previous center.*  

    *     centers_dist: distance matrix between centers.*  

    */
    //若迭代次数为1 返回初始质心关于自身的距离矩阵
    if (iteration_counter == 1)
        centers_dis = cal_dist(new_centroids, new_centroids).array().sqrt(); //假如new_centroids为m*n的矩阵，则返回的是关于自身的m*m的距离矩阵
    //若迭代次数大于1 考虑缩减质心-质心距离计算
    else
    {
        for (unsigned int i = 0; i < k; i++)
        {
            for (unsigned int j = 0; j < k; j++)
            {
                //满足缩减计算的条件 说明簇i和簇j不是近邻 给一个距离值
                //注：当k较大的时候，这一步可以减少计算量；赋值前，centers_dis表示上一轮迭代的距离矩阵，赋值后表示本次迭代更新后的距离矩阵
                if (centers_dis(i, j) >= 2 * the_rs(i) + delta(i) + delta(j))
                    //由于centers_dis(i, j) - delta(i) - delta(j)≥2*thes_rs(i) 此时簇j不是i的近邻 因此可以指定它们两个的距离就为centers_dis(i, j) - delta(i) - delta(j)
                    //这个公式实际将质心-质心的距离缩小 若缩小后的距离依然没有近邻 则按原始的更大距离计算的结果也是没有近邻 故可以用`centers_dis(i, j) - delta(i) - delta(j)`近似刻画j不是i的近邻时的质心-质心距离
                    centers_dis(i, j) = centers_dis(i, j) - delta(i) - delta(j);
                //不满足缩减计算的条件 直接计算欧氏距离
                else
                {
                    centers_dis(i, j) = sqrt((new_centroids.row(i) - new_centroids.row(j)).squaredNorm());
                }
            }
        }
    }
}

inline MatrixOur cal_dist(MatrixOur &dataset, MatrixOur &centroids)
{

    /*

    *Summary: Calculate distance matrix between dataset and center point

    *Parameters:

    *     dataset: dataset matrix.*   

    *     centroids: centroids matrix.* 

    *Return : distance matrix between dataset and center point.

    *注意：
    - 若dataset为m*k的矩阵，centroid为n*k的矩阵，则返回的是m*n的矩阵，其中每个元素表示dataset对应元素与centroid对应元素的欧式距离的平方值（是的！这里有平方）
    - 这里的代码采用的计算公式是经过推导设计的，应该有助于加速计算
    - 公式可分为三部分理解：
        1. `-2 * dataset * (centroids.transpose())).colwise()`： 一个m*n的矩阵 | Python版本：-2*dataset*centroids.T
        2. `dataset.rowwise().squaredNorm()).rowwise()`：一个m*1的矩阵 | Python版本：dataset中行元素求内积
        3. `centroids.rowwise().squaredNorm()).transpose()`： 一个1*n的矩阵 | Python版本：centroids行元素求内积后再转置
        后面两部分直接加到第一部分的矩阵上会发生传播
    */

    return (((-2 * dataset * (centroids.transpose())).colwise() + dataset.rowwise().squaredNorm()).rowwise() +
            (centroids.rowwise().squaredNorm()).transpose());
}

inline MatrixOur
cal_ring_dist_Ring(unsigned j, unsigned int data_num, unsigned int dataset_cols, MatrixOur &dataset,
                   MatrixOur &now_centers,
                   ClusterIndexVector &now_data_index)
{

    /*

    *Summary: 计算环域的中点到其最近邻簇之间的距离矩阵
    Calculate the distance matrix from the point in the ring area to the corresponding nearest neighbor

    *Parameters:

    *     j: the label of the current ring.* 

    *     data_num: number of points in the ring area.*   

    *     dataset_cols: data set dimensions.* 

    *     dataset: dataset in eigen matrix format.* 

    *     now_centers: nearest ball center matrix corresponding to the current ball.* 

    *     now_data_index: labels for points in each ring.* 

    *Return : distance matrix from the point in the ring area to the corresponding nearest neighbor.

    假如当前环域是第j层（j从1开始计）环域 该环域有n个点 则返回的是n*(j+1)的距离矩阵
    列数为j+1是因为考虑了当前簇自身
    */

    MatrixOur data_in_area(data_num, dataset_cols);

    for (unsigned int i = 0; i < data_num; i++)
    {
        data_in_area.row(i) = dataset.row(now_data_index[j - 1][i]);
    }

    Ref<MatrixOur> centers_to_cal(now_centers.topRows(j + 1));

    return (((-2 * data_in_area * (centers_to_cal.transpose())).colwise() +
             data_in_area.rowwise().squaredNorm())
                .rowwise() +
            (centers_to_cal.rowwise().squaredNorm()).transpose());
}

inline MatrixOur
cal_ring_dist_noRing(unsigned int data_num, unsigned int dataset_cols, MatrixOur &dataset, MatrixOur &now_centers,
                     vector<unsigned int> &now_data_index)
{
    /*

    *Summary: 计算环域中的点到其最近的近邻簇的距离 Calculate the distance matrix from the point in the ring area to the corresponding nearest neighbor

    *Parameters:

    *     data_num: number of points in the ring area.*   

    *     dataset_cols: data set dimensions.* 

    *     dataset: dataset in eigen matrix format.* 

    *     now_centers: nearest ball center matrix corresponding to the current ball.* 

    *     now_data_index: labels for points in the ring.* 

    *Return : 返回环域中点到所有近邻簇的距离矩阵 distance matrix from the point in the ring area to the corresponding nearest neighbor.
    
    * 注：
        1. 假如 环域中的点的坐标矩阵（now_centers）形状是(m,dataset_cols)、所有近邻簇（包含当前簇）的坐标矩阵形状是(n,dataset_cols) 
        那么返回的距离矩阵的形状是(m,n)
        2. 返回的距离矩阵中的距离实际上是欧式距离的平方 

    */
    //临时变量 二维矩阵 存储当前簇中环域的所有点的坐标
    MatrixOur data_in_area(data_num, dataset_cols);

    for (unsigned int i = 0; i < data_num; i++)
    {
        data_in_area.row(i) = dataset.row(now_data_index[i]);
    }

    return (((-2 * data_in_area * (now_centers.transpose())).colwise() +
             data_in_area.rowwise().squaredNorm())
                .rowwise() +
            (now_centers.rowwise().squaredNorm()).transpose());
}

void initialize(MatrixOur &dataset, MatrixOur &centroids, VectorXi &labels, ClusterIndexVector &cluster_point_index,
                ClusterIndexVector &clusters_neighbors_index, ClusterDistVector &temp_dis)
{

    /*

    *Summary: Initialize related variables

    *Parameters:

    *     dataset: dataset in eigen matrix format.*   

    *     centroids: dcentroids matrix.* 

    *     labels: the label of the cluster where each data is located.* 

    *     clusters_point_index: two-dimensional vector of data point labels within each cluster.一个二维向量 每个一维向量元素表示一个簇内的所有样本的索引* 

    *     clusters_neighbors_index: two-dimensional vector of neighbor cluster labels for each cluster.一个二维向量 每个一维向量元素表示一个簇的所有近邻簇的索引* 

    *     point_center_dist/temp_dis: distance from point in cluster to center.一个二维向量 每个一维向量元素表示一个簇中所有点到簇中心的距离* 

    */

    MatrixOur::Index minCol;
    for (int i = 0; i < centroids.rows(); i++)
    {
        cluster_point_index.push_back(vector<unsigned int>()); // push_back: 在vector末尾追加元素
        clusters_neighbors_index.push_back(vector<unsigned int>());
        temp_dis.push_back(vector<OurType>());
    }

    // //【测试】保存temp_dis向量
    // std::ofstream outFile;
    // outFile.open("/home/ubuntu/workspace/python/luyao/experiment/ball-k-means-master/C++Version/temp_dis.txt");
    // outFile << temp_dis;
    // outFile.close();

    MatrixOur M = cal_dist(dataset, centroids); //二维向量 每个一维向量元素表示每个样本点到所有初始中心的距离
    for (int i = 0; i < dataset.rows(); i++)
    {
        M.row(i).minCoeff(&minCol);               //返回每个样本点到最近初始质心的初始中心的索引
        labels(i) = minCol;                       //记录第i个样本点距离最近质心的索引
        cluster_point_index[minCol].push_back(i); //将第i个样本分配到距离最近的簇
    }
}

inline MatrixOur initial_centroids(MatrixOur dataset, int k, int random_seed = -1)
{
    int dataset_cols = dataset.cols();
    int dataset_rows = dataset.rows();
    vector<int> flag(dataset_rows, 0);

    MatrixOur centroids(k, dataset_cols);
    int initial_row = 0;
    if (random_seed == -1)
    {
        srand((unsigned)time(NULL));
        initial_row = rand() % dataset_rows;
    }
    else
    {
        initial_row = dataset_rows % random_seed; //random_seed不能为0 这里有个异常
        srand(random_seed);
    }
    //将原始数据集中的第initial_row行数据作为一个初始质心
    centroids.row(0) = dataset.row(initial_row);
    flag[initial_row] = 1;

    vector<OurType> nearest(dataset_rows, 0);

    OurType t_dist = 0;

    for (int i = 0; i < k - 1; i++)
    {
        vector<OurType> p(dataset_rows, 0);

        for (int j = 0; j < dataset_rows; j++)
        {
            //计算质心向量中当前质心与所有原始样本点的距离
            t_dist = sqrt((dataset.row(j) - centroids.row(i)).squaredNorm());
            if (i == 0 && flag[j] != 1)
                nearest[j] = t_dist;
            else if (t_dist < nearest[j])
                nearest[j] = t_dist;

            if (j == 0)
                p[j] = nearest[j];
            else
                p[j] = p[j - 1] + nearest[j];
        }

        OurType rand_num = rand() % 1000000001;
        rand_num /= 1000000000;

        for (int j = 0; j < dataset_rows; j++)
        {
            p[j] = p[j] / p[dataset_rows - 1];
            if (rand_num < p[j])
            {
                centroids.row(i + 1) = dataset.row(j);
                flag[j] = 1;
                nearest[j] = 0;
                break;
            }
        }
    }
    return centroids;
}

VectorXi ball_k_means(MatrixOur &dataset, int k, bool isRing = false, bool detail = false,
                      int random_seed = -1, const char *filename = "0")
{
    /**粒球kmeans函数主体
     * datasset：MatrixOur 数据集
     * k：int 簇数
     * isRing：bool 是否为含有ring的算法
     * detail：bool 是否返回详细的中间内容
     * random_seed：int
     * filename：char* 聚类初始中心的文件句柄
     */
    MatrixOur centroids;
    if (filename == "0")
    {
        centroids = initial_centroids(dataset, k, random_seed);
    }
    else
    {
        centroids = load_data(filename);
    }

    // //【测试】指定random_seed不为-1 每次质心生成的是否一致
    // outFile.open("/home/ubuntu/workspace/python/luyao/experiment/ball-k-means-master/C++Version/centroids2.txt");
    // outFile << centroids;
    // outFile.close();

    VectorXi labels;
    //isRing=true表示选择的算法实现考虑将环域进一步划分（第1,2,3...环域） 否则表示仅仅将簇分为稳定域和环域
    if (isRing)
    {
        labels = ball_k_means_Ring(dataset, centroids, detail);
    }
    else
    {
        labels = ball_k_means_noRing(dataset, centroids, detail);
    }
    return labels;
}

int main(int argc, char *argv[])
{
    double start_time, end_time;
    double all_start_time, all_end_time;

    all_end_time = clock();
    //读取数据集
    cout << "start to load dataset..\n";
    start_time = clock();
    MatrixOur dataset = load_data("../data+centers/dataset/Epileptic.csv");
    // MatrixOur dataset = load_data("../PythonVersion/data.csv"); //【测试】验证算法有效性 随机生成数据集测试
    end_time = clock();
    cout << "load dataset over!\n";
    cout << "[Time]load dataset costs " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "[s]" << endl;

    //开始训练
    cout << "start to fit..\n";
    start_time = clock();
    // 簇数
    int k = 50;
    // 是否对环域细划分
    bool isRing = true;
    // 是否输出中间信息
    bool detail = true;
    // 随机数种子 若 random_seed==-1 则每次生成的初始质心都一样 注：该参数不能为0 否则会引发除0异常！！
    int random_seed = -1;
    // 初始质心的文件名 若 filename ="0" 则表示使用程序随机生成的质心 此时random_seed参数的值才有意义
    const char *filename = "0";
    VectorXi labels = ball_k_means(dataset, k, isRing, detail, random_seed, filename);
    end_time = clock();
    cout << "fit over!\n";
    cout << "[Time]fit costs " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "[s]" << endl;

    //保存结果：把点的簇标签保存到文件
    cout << "start to save results..\n";
    start_time = clock();
    // std::ofstream outFile;
    outFile.open("/home/ubuntu/workspace/python/luyao/experiment/ball-k-means-master/C++Version/results.txt");
    // outFile.open("../PythonVersion/test_results.txt"); //【测试】验证算法有效性 数据集的结果
    outFile << labels;
    outFile.close();
    end_time = clock();
    cout << "save results over!\n";
    cout << "[Time]save results costs " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "[s]" << endl;

    all_end_time = clock();
    cout << "[Time]all steps costs " << (double)(all_end_time - all_start_time) / CLOCKS_PER_SEC << "[s]" << endl;
}
