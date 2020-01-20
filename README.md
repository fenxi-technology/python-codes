# python-codes
Some general functions&amp;LSTM model


Main_LSTM.py

LSTM的类，过程分为scale-feature selection-split data-train LSTM model-test model-inverse scale-plot。
其中在feature selection中有4个选项：通过correlation/random forest/recursive feature elimination/pca进行特征选择，
前三项会input k值，pca使用方差百分比作为输入值。
在split data这一步中，function需要输入look_back和lead_time值，look_back代表会被放入模型中的时间序列的长度，
lead_time代表提前多长时间进行预测。
还有一些LSTM模型的参数包括每层的激活函数（activation function）、层数(LSTM层，压缩层)、drop_out比例（随机drop
神经元来防止overfit）、所有数据的循环次数（epoch）、每次放入模型的小部分数据量（batch_size）、loss函数的选择、
optimizer的选择。


class_moving_series.py

创建移动特征的类。其中包括: 创建移动相关值、移动平均、移动最小值、移动最大值、移动中间值、移动方差、移动峰值、
移动倾斜度、指数移动平均、指数移动方差、二次指数移动平均。所有选项均可选择。
函数需要input的值分别有dataframe（数据），y_col（即需要预测的列），window list，alpha list，
beta list，correlation coefficient（目前使用0.5）和minimu; periods。
在创建所有移动值（不包括指数和二次指数）时，函数会创建每个数据周围window_size个值的平均/最大/...等等值，window list包
含所有会使用的window_size值，例如window list=[5,10,15]则会创建window_size=5，10，15的平均/最大/..等等值，min
periods则代表需要有几个非NAN值才可计算。
在创建移动相关值时，函数会分别计算数据内所有其他列和预测值列的pearson correlation, 并选取结果大于我们的设定的
correlation coefficient的列进行移动相关值的创建（在例子里是导电率和ph值）。此时包括自相关因为自相关的correlation=1。
在创建指数移动平均和二次指数移动平均时，函数会使用alpha值和beta值作为平滑参数平衡历史数据，取值属于（0，1）。
alpha和beta越小，则数据平滑力度越大。同样输入的alpha和beta值属于list，可输入多个。


FIND PERIOD.py

在寻找period之前，尝试了对数据进行平滑处理，使用了3种方法：
1. 计算每个数据点前后的差分，并对差分进行统计学的计算，把1.5倍四分位距之外的点用整个数据的中位值替换。
2. 计算每个数据点前后的差分，并对差分进行统计学的计算，把1.5倍四分位距之外的点用前后值（范围可选）均匀填充。
3. 使用之前代码里提到的移动平均。
在平滑结束后，手动输入周期性T（类似5，10，20）后把我们的数据分为T组，计算分别的欧式距离并求得平均。
平滑处理的3种方法发现第三种速度较快且效果较好，第二种原理上可行，但是有2个循环速度太慢了。最后寻找周期性的
结果很差，发现数值越大距离（差别）最大。


find_window_size.py
在这个函数里首先进行时间转换；其次写了一个函数选取自方差大于threshold值（可选，目前使用0.4）的列作为possible
feature columns（首轮筛选）。这个函数同样也用在class_moving_series种，即选取用来创建移动值的特征列。
接下来后函数使用之前创建的class_moving_series类和class_time_features类创建了时间特征和基于windows的
移动特征。windows list在这个函数里为20到1440（1440为一天的分钟值），中间间隔50，即20，70，120...1440。
开始为20可以保证不会有太多的无效值，间隔为50考虑到电脑的内存跑不动那么多特征（试过10，20的间隔失败了）。
函数使用了4种方法进行特征选择：
1. 使用selectKbest选取k个correlation相对高的列。
2. 使用correlation matrix对correlation进行排序，选取k个相对高的列，同第一种方法但是跑的很慢。
3. 使用递归特征消除法，并使用线性回归作为基础模型选取k个特征。这个方法的问题在于线性回归比较适合基础数据，
对我们的数据相性不好。
4. 使用random forest对数据建模，并返回importance列，选取k个importance高的特征列。
最后把这个函数整理了一下放入了main_LSTM。


influxdb.py
连接Python和influxdb。实现的功能有：把数据传入influxdb，influxdb取数据，查询数据，把influxdb里的数据
格式改成python数据分析需要的数据格式。
其中query可以输入指令对influxdb的数据进行查询，query_df返回查询后的dataframe，query_all_df返回查询后的measurement
的所有数据，insert_points插入单个数据，insert_df插入整个dataframe，show_all_tables，drop_tables都如其名。
influxdb_df_to_python_df_update返回python数据分析需要的数据格式。


class explorative analysis.py
Input dataframe, object list, a list of columns we would like to explore and correlation coefficient.
其中有function：
1.convert dtypes把dtypes从object变为float或者0/1的boolean
2.check missing返回missing values概况和处理方式。如果最大null值超过数据大小的0.05倍，则删除null值；
反之用均值或者众数填补。整列全为null值的特征栏直接删除
3.correlation_matrix返回correlation matrix
4.check_correlation返回input的columns与其他特征的相关值>coefficient值的排列。
5. plot_df_correlation_full返回所有特征列的correlation画图。
6. plot_df_correlaiton_use返回不全为同一个值的特征列（即variance不等于0）的correlation画图。
7. plot_columns_freq返回input的columns的frequency图。
8.plot_columns_dist返回input的columns的根据时间上的轨迹图。
9. column_describe返回input的columns的概况。


correlation between.py
input表格名，目标列，和coefficient。
返回表格中的目标列和其他所有表格（不包括电表）的大于coefficient的相关值。

