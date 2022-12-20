# coding=utf-8
from flask import Flask, request, render_template, redirect, url_for
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from efficient_apriori import apriori
from werkzeug.serving import run_simple
from abc import ABCMeta
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64, traceback, sqlite3, time, random


app = Flask(__name__,template_folder='templates')


class MyDB:
    def __init__(self):
        try:
            self.conn = sqlite3.connect('data_management.db')
        except Exception as e:
            print("database connection failed：%s" % e)

    def close(self):
        self.conn.close()

    def get_cursor(self):
        return self.conn.cursor()

    def get_connection(self):
        return self.conn

    def tbl_opt(self, sql):
        flag = 1
        try:
            cursor = self.get_cursor()
            cursor.execute(sql)
            print("opt table successfully.")
            self.conn.commit()
        except Exception as e:
            print(e)
            flag = 0
            self.conn.rollback()
            traceback.print_exc()
            print('traceback.format_exc():\n%s' % traceback.format_exc())
        return flag

    def execute_one(self,sql):
        try:
            cur = self.get_cursor()
            cur.execute(sql)
            self.conn.commit()
            return cur.rowcount, cur.fetchall()
        except Exception as e:
            msg = "database err"
            self.conn.rollback()
            print(e)
            traceback.print_exc()
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0, []

    @staticmethod
    def get_create_sql_manager():
        sql_create_tbl = """
        CREATE TABLE if not exists manager_info (
            manager_id INT NOT NULL PRIMARY KEY UNIQUE,
            password VARCHAR(20) NOT NULL, 
            info TEXT,
            create_time INT,
            is_super BOOLEAN
        )
        """
        return sql_create_tbl

    @staticmethod
    def get_create_sql_item_click():
        sql_create_tbl = """
        CREATE TABLE if not exists item_click (
            user_id INT NOT NULL,
            item_id INT NOT NULL, 
            click_date DATE,
            create_time INT
        )
        """
        return sql_create_tbl

    @staticmethod
    def get_create_sql_item_info():
        sql_create_tbl = """
        CREATE TABLE if not exists item_info (
            item_id INT NOT NULL PRIMARY KEY UNIQUE, 
            name VARCHAR(200) NOT NULL,
            create_time INT,
            item_type VARCHAR(50)
        )
        """
        return sql_create_tbl

    @staticmethod
    def get_create_sql_user_info():
        sql_create_tbl = """
        CREATE TABLE if not exists user_info (
            user_id INT NOT NULL PRIMARY KEY UNIQUE, 
            name VARCHAR(100) NOT NULL,
            password VARCHAR(20) NOT NULL, 
            create_time INT,
            gender VARCHAR(7)
        )
        """
        return sql_create_tbl

    @staticmethod
    def get_create_sql_user_stat():
        sql_create_tbl = """
        CREATE TABLE if not exists user_stat (
            user_id INT PRIMARY KEY UNIQUE NOT NULL,
            feature1 FLOAT,
            feature2 FLOAT,
            feature3 FLOAT,
            feature4 FLOAT,
            feature5 FLOAT,
            feature6 FLOAT,
            feature7 FLOAT,
            feature8 FLOAT,
            feature9 FLOAT,
            feature10 FLOAT,
            last_update_time INT NOT NULL
        )
        """
        return sql_create_tbl


    @staticmethod
    def init_manager():
        ts = time.time()
        return 'INSERT INTO manager_info (manager_id, password, info,create_time, is_super)' \
               ' VALUES (\'admin\',\'123\', \'the boss\',%d, 1)' % time.time()

    @staticmethod
    def get_drop_sql():
        return "drop table if exists students"


class User(metaclass=ABCMeta):
    def __init__(self,username,password='', txt='',role=''):
        self.username = username
        self.password = password
        self.ts = time.time()
        self.role = role
        self.info = txt

    def __str__(self):
        return self.username

    def register(self, repassword):
        msg = ''
        db = MyDB()
        sql_select = "SELECT * FROM manager_info where manager_id=\'%s\'" % self.username
        sql_insert = """INSERT INTO manager_info (manager_id, password, info,create_time, is_super) 
                            VALUES (\'{uname}\',\'{pwd}\', \'{info}\',\'{ts}\',{flag})""".format(
            uname=self.username,
            pwd=self.password,
            info=self.info,
            ts=self.ts,
            flag=0)

        if self.password == repassword:
            cnt_select, rows = db.execute_one(sql_select)
            if len(rows) == 1:
                return render_template('register.html', msg='user already exists')
            elif len(rows) == 0:
                cnt_insert, rows = db.execute_one(sql_insert)
                print(sql_insert)
                print(cnt_insert, len(rows))
                if cnt_insert > 0:
                    msg = "Adding data successfully"
                else:
                    msg = 'Failed to add the data'
            else:
                db.conn.rollback()
                msg = "database err"

        db.close()
        return msg

    def login_check(self):
        db = MyDB()
        sql = "SELECT * FROM manager_info " \
              "WHERE manager_id=\'%s\' AND password=\'%s\'" \
              % (self.username, self.password)
        cnt, rows= db.execute_one(sql)
        db.close()
        print(cnt,len(rows))
        if len(rows) == 1:
            print('login_row4:',rows[0][4])
            return rows[0][4]
        else:
            return None


class Manager(User):
    def __init__(self, username, password='', txt='', role=''):
        super(Manager,self).__init__(username, password='', txt='', role='')
        self.level = 1

    def delete(self):
        db = MyDB()
        sql = "DELETE FROM manager_info " \
              "WHERE manager_id=\'%s\' and is_super=0" % self.username
        cnt, rows = db.execute_one(sql)
        if cnt > 0:
            msg = "Data Deleted Successfully"
        else:
            msg = 'Failed to delete the data'
        db.close()
        return msg


def csv2sql_user():
    tripadvisor = pd.read_csv('data/tripadvisor.csv')
    tripadvisor = tripadvisor.rename(columns=dict(map(lambda x: (x, x.replace(' ', '_')),
                                                      list(tripadvisor.keys())
                                                      )
                                                  )
                                     )

    my_db = MyDB()
    conn = my_db.get_connection()
    cur = conn.cursor()
    sql_stat = """
        INSERT INTO user_stat (user_id, feature1,feature2,feature3,feature4,feature5,
            feature6,feature7,feature8,feature9,feature10,last_update_time)
        VALUES (%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d)
    """

    sql_info = """
        INSERT INTO user_info (user_id, name, password, create_time, gender)
        VALUES (%d,\'%s\',\'123\',%d,\'%s\')
    """
    gender = ('male','female','unknown')
    for i, r in tripadvisor.iterrows():
        uid = int(r[0].split(' ')[-1])
        ts = time.time()
        idx = random.randint(0,2)
        sql_stat4insert = sql_stat % \
                          (uid,r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9],r[10], ts)
        sql_info4insert = sql_info % (uid, r[0].replace(' ', "_"),ts,gender[idx])

        cur.execute(sql_info4insert)
        cur.execute(sql_stat4insert)
    conn.commit()
    conn.close()


def csv2sql_items():
    my_db = MyDB()
    conn = my_db.get_connection()
    cur = conn.cursor()
    sql_mk_item_idx = """CREATE INDEX IF NOT EXISTS 
                            idx_item_click ON item_click(user_id,item_id)"""
    sql_stat = """
        INSERT INTO item_click (item_id, user_id, click_date, create_time)
            VALUES ({item_id},{user_id},\'{click_date}\',{create_time})
    """
    sql_info = """
            INSERT INTO item_info (item_id,name,create_time,item_type)
                SELECT {item_id}, \'{item_name}\', {create_time}, \'{item_type}\'
            WHERE NOT EXISTS (
                SELECT * FROM item_info 
                WHERE item_id={id_on_where}
            )
    """

    groceries = pd.read_csv('data/Groceries_dataset.csv', sep=',')
    tdict = {}
    for idx, tname in enumerate(set(groceries['itemDescription'])):
        tdict[tname] = idx

    for i, r in groceries.iterrows():
        ts = time.time()
        uid = int(r[0]) if int(r[0])<4000 else int(r[0])-4001
        tid = tdict[r[2]]
        tname = r[2].replace(' ', '_').replace("/", '_').replace('.','').replace('-','_')\
            .replace('(', "").replace(')', "")
        date = r[1]
        if tname in ['newspapers', 'margarine']:
            ttype = 'readings'
        else:
            ttype = tname.split('_')[-1]
        sql_stat4insert = sql_stat.format(item_id=tid,
                                          user_id=uid,
                                          click_date=date,
                                          create_time=ts)
        sql_info4insert = sql_info.format(item_id=tid,
                                          item_name=tname,
                                          create_time=ts,
                                          item_type=ttype,
                                          id_on_where=tid)
        cur.execute(sql_mk_item_idx)
        cur.execute(sql_stat4insert)
        cur.execute(sql_info4insert)
    conn.commit()
    conn.close()


def sql_init():
    my_db = MyDB()
    # my_db.tbl_opt(my_db.get_create_sql_manager())
    # my_db.execute_one(my_db.init_manager())

    my_db.tbl_opt(my_db.get_create_sql_item_click())
    my_db.tbl_opt(my_db.get_create_sql_item_info())
    csv2sql_items()

    # my_db.tbl_opt(my_db.get_create_sql_user_stat())
    # my_db.tbl_opt(my_db.get_create_sql_user_info())
    # csv2sql_user()
    my_db.close()


def img_user_gender_pie():
    my_db=MyDB()
    conn = my_db.get_connection()
    cur = conn.cursor()
    sql_gender = """
        SELECT gender, COUNT(gender) 
        FROM user_info GROUP BY gender 
    """
    cur.execute(sql_gender)
    x = []
    labels = []
    for r in cur.fetchall():
        x.append(r[1])
        if r[0] != 'nuknown':
            labels.append(r[0])
        else:
            labels.append('unknown')
    plt.pie(x=x, labels=labels,autopct='%.2f%%')
    plt.title("User Gender Percentage")
    plt.ion()

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims
    plt.clf()
    plt.close()
    return imd


def pca4chart():
    my_db=MyDB()
    conn = my_db.get_connection()
    cur = conn.cursor()
    sql_user_all = """
        SELECT * from user_stat
    """
    cur.execute(sql_user_all)
    mtx = []
    for r in cur.fetchall():
        lst = list(r)[1:-1]
        mtx.append(lst)
    conn.commit()
    conn.close()
    pca = PCA()
    pca.fit(mtx)
    n_pcs = pca.components_.shape[0]
    pca_ratio = pca.explained_variance_ratio_
    important_ratio_bottom = []
    important_ratio_height = []
    feat_ratio = []
    feature_acc = [0] * n_pcs

    for j, i in enumerate(range(n_pcs)):
        component_arr = np.abs(pca.components_[i])
        idx = component_arr.argmax()
        component_ratio = component_arr[idx] / sum(component_arr)
        feat_ratio.append("component_"+str(j+1)+'    feature_'
                          +str(idx+1)+'    '+str('%.5f' % component_ratio))
        important_ratio_bottom.append(pca_ratio[i] * component_ratio)
        important_ratio_height.append(pca_ratio[i] * (1 - component_ratio))
        for k, v in enumerate(component_arr):
            feature_acc[k] += v
    return pca_ratio, important_ratio_bottom, important_ratio_height, feature_acc, feat_ratio


def img_pca_components_importance(tup):
    pca_ratio = tup[0]
    important_ratio_bottom = tup[1]
    important_ratio_height = tup[2]
    plt.bar(x=list(range(1, len(pca_ratio) + 1)),
            bottom=important_ratio_bottom,
            height=important_ratio_height,

            )
    plt.bar(x=list(range(1, len(pca_ratio) + 1)),
            height=important_ratio_bottom,
            label='Rotio of the most important feature'
            )
    plt.xlabel('Components NO')
    plt.ylabel('Ratio of Importance')
    plt.xticks(range(1, len(pca_ratio) + 1))
    plt.legend()

    plt.ion()
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims
    plt.clf()
    plt.close()
    return imd


def img_pca_accumulation_importance(tup):
    pca_ratio = tup[0]
    feature_acc = tup[3]
    plt.bar(x=list(range(1, len(pca_ratio) + 1)),
            height=feature_acc
            )
    plt.xlabel('Feature NO')
    plt.ylabel('Accumulation of Feature Importance')
    plt.xticks(range(1, len(pca_ratio) + 1))

    plt.ion()
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims
    plt.clf()
    plt.close()
    return imd


def my_kmeans(data_df, k):
    my_kmeans = KMeans(n_clusters=k, random_state=1)
    my_kmeans.fit(data_df)
    clusters = my_kmeans.predict(data_df)
    centroids = my_kmeans.cluster_centers_
    return clusters, pd.DataFrame(centroids)


def img_kmeans_silhouette(feats_str):
    print('feat_drop_lst in img_kmeans_silhouette:',feats_str)
    feat_drop_lst = feats_str.split(',')
    string = ""
    for i in range(1,10):
        tmp = 'feature'+str(i)
        if tmp in feat_drop_lst:
            continue
        string += ","+tmp
    string = string[1:]

    conn = MyDB().get_connection()
    cur = conn.cursor()
    sql =  'SELECT %s FROM user_stat' % string
    cur.execute(sql)
    mtx = []
    for r in cur.fetchall():
        mtx.append(r)
    pca_df = pd.DataFrame(mtx)
    sil_score_lst = []
    k_lst = list(range(2, 20))
    for k in k_lst:
        clusters, centroids_df = my_kmeans(pca_df, k)
        score = silhouette_score(pca_df, clusters, metric='euclidean')
        sil_score_lst.append(score)

    plt.plot(k_lst, sil_score_lst)
    plt.title('Optimal number of clusters')
    plt.scatter(k_lst, sil_score_lst, marker='x', color='red')
    plt.ylabel('sil_score')
    plt.xticks(k_lst)

    plt.ion()
    # figure 保存为二进制文件
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    # 将matplotlib图片转换为HTML
    imb = base64.b64encode(plot_data)  # 对plot_data进行编码
    ims = imb.decode()
    imd = "data:image/png;base64," + ims
    plt.clf()
    plt.close()
    return imd


def img3d_kmeans(feats_str):
    print('feat_drop_lst in img_kmeans_silhouette:',feats_str)
    feat_drop_lst = feats_str.split(',')
    string = ""
    for i in range(1,10):
        tmp = 'feature'+str(i)
        if tmp in feat_drop_lst:
            continue
        string += ","+tmp
    string = string[1:]

    conn = MyDB().get_connection()
    cur = conn.cursor()
    sql =  'SELECT %s FROM user_stat' % string
    cur.execute(sql)
    mtx = []
    for r in cur.fetchall():
        mtx.append(r)
    pca_df = pd.DataFrame(mtx)
    clusters, centroids_df = my_kmeans(pca_df, 4)
    pca_df['cluster_label'] = clusters

    fig = plt.figure()
    colors = ['r', 'g', 'b', 'yellow']
    markers = ['+', '^', 'o', '*']
    ax = fig.add_subplot(111, projection='3d')
    for i in clusters:
        tmp_df = pca_df[pca_df['cluster_label'] == i]
        xs = np.array(tmp_df[2])
        ys = np.array(tmp_df[5])
        zs = np.array(tmp_df[1])
        ax.scatter(xs, ys, zs, c=colors[i], marker=markers[i])
    ax.set_xlabel('Feature3')
    ax.set_ylabel('Feature6')
    ax.set_zlabel('Feature2')

    plt.ion()
    # figure 保存为二进制文件
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    # 将matplotlib图片转换为HTML
    imb = base64.b64encode(plot_data)  # 对plot_data进行编码
    ims = imb.decode()
    imd = "data:image/png;base64," + ims
    plt.clf()
    plt.close()
    return imd


@app.route('/show')
def user_center():
    users = []
    db = MyDB()
    sql = 'SELECT * FROM manager_info'
    cnt, rows = db.execute_one(sql)
    for r in rows:
        if r[4] == 1:
            role = "super"
        else:
            role = 'normal'
        print(r[0],r[1],r[2])
        users.append(User(r[0],r[1],r[2], role))
    db.close()
    return render_template('/show.html',users=users)


@app.route('/register',methods =['GET','POST'])
def register():
    msg = ''
    if request.method == 'POST':
        username = request.form.get('username')
        info = request.form.get('info')
        password = request.form.get('password')
        repassword = request.form.get('repassword')
        user = User(username, password, info)
        msg = user.register(repassword)
    return render_template('register.html', msg=msg)


@app.route('/',methods =['GET','POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        username = request.form.get('username')
        password = request.form.get('password')
        user = User(username, password)
        flag = user.login_check()
        if flag is None:
            return redirect('/')
        else:
            if flag == 1:
                return redirect('/super')
            else:
                return redirect('/normal')


@app.route('/del')
def del_user():
    username = request.args.get('username')
    mg = Manager(username)
    msg = mg.delete()
    return render_template('/info.html',msg=msg)


@app.route('/super')
def front_page():
    return render_template('frontpage.html')


@app.route('/normal')
def front_page2():
    return render_template('frontpage2.html')


@app.route('/secpage')
def secpage():
    return render_template('secpage.html')


@app.route('/thrdpage')
def thrdpage():
    return render_template('thrdpage.html')


tup = pca4chart()


@app.route('/visual/?<string:feats>')
def visual(feats):
    global tup
    if feats == 'stat':
        imd1 = img_user_gender_pie()
        imd2 = None
    elif feats == 'pca':
        imd1 = img_pca_accumulation_importance(tup)
        imd2 = img_pca_components_importance(tup)
    elif feats == 'pca_feat':
        return render_template('/blank.html', pca_feat_lst=tup[4])
    else:
        print(feats)
        imd1 = img_kmeans_silhouette(feats)
        imd2 = img3d_kmeans(feats)
    return render_template('/visual.html', img=imd1,img2=imd2)


@app.route("/feat",methods =['GET','POST'])
def feat():
    if request.method == 'GET':
        feats = []
        mydb = MyDB()
        sql = 'SELECT name FROM pragma_table_info(\'user_stat\')'
        conn = mydb.get_connection()
        cur = conn.cursor()
        cur.execute(sql)
        for r in cur.fetchall():
            if r[0] == 'last_update_time' or r[0] == 'user_id':
                continue
            feats.append(r[0])
        conn.commit()
        conn.close()
        return render_template('feat.html', feats=feats)
    else:
        feat_drop_lst = request.form.getlist('feat')
        feats_str = ','.join(feat_drop_lst)
    return redirect(url_for('visual', feats=feats_str))


@app.route("/param",methods =['GET','POST'])
def param():
    if request.method == 'GET':
        return render_template('param.html')
    else:
        min_confidence = float(request.form.get('min_confidence'))
        min_support = float(request.form.get('min_support'))
        if 0 <= min_confidence <= 1 and 0 <= min_support <= 1:
            params = str(min_confidence)+','+str(min_support)
            print(params)
            return redirect(url_for('apriori_rules', params=params))
        else:
            return render_template('param.html')


@app.route('/apriori_rules/?<string:params>')
def apriori_rules(params):
    min_confidence = float(params.split(',')[0])
    min_support = float(params.split(',')[1])
    conn = MyDB().get_connection()
    cur = conn.cursor()
    sql_join = """
        SELECT CAST(a.user_id AS STRING),
            b.name
        FROM item_click a 
        LEFT JOIN item_info b 
            ON a.item_id=b.item_id
    """
    cur.execute(sql_join)
    mtx = cur.fetchall()
    groceries = pd.DataFrame(mtx)
    groceries_grouped = groceries.groupby(0)[1] \
        .agg(lambda x: ','.join(x)) \
        .reset_index(name='Item_series')
    input_items = list(map(lambda x: tuple(set(x.split(','))),
                           list(groceries_grouped['Item_series'])
                           ))
    item_dict, associated_rules = apriori(input_items,
                                          min_support=min_support,
                                          min_confidence=min_confidence)

    reflect_rec = list((map(lambda x: (x.lhs[0], x.rhs[0]), associated_rules)))
    dict = {}
    for tup in reflect_rec:
        if tup[0] not in dict:
            dict[tup[0]] = tup[1]
        else:
            dict[tup[0]] += ', ' + tup[1]
    strings = []
    for k in dict:
        strings.append(k+": "+dict[k])
    return render_template('/apriori.html', params=strings)


if __name__ == '__main__':
    # sql_init()
    run_simple('localhost', 9001, app)
