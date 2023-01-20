# coding=utf-8
from flask import Flask, request, render_template, redirect, url_for,session
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

from sklearn.preprocessing import StandardScaler
import hashlib, os
from werkzeug.utils import secure_filename
from plotly import graph_objects as go
from pyecharts.charts import Radar
from pyecharts import options as opts
import plotly.express as px
from pyecharts.charts import Funnel



app = Flask(__name__,template_folder='templates')
app.secret_key = 'random string'
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['jpeg', 'jpg', 'png', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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


###########################################  above by Yiqing ###########################################################
###########################################  follow by Ting Wei ##########################################################

def getLoginDetails():
    with sqlite3.connect('data_management.db') as conn:
        cur = conn.cursor()
        if 'email' not in session:
            loggedIn = False
            name = ''
            noOfItems = 0
            noOfOrderss = 0
            noOfFavorite = 0
        else:
            loggedIn = True
            cur.execute("SELECT user_id,name FROM user_info WHERE email=?", (session['email'],))
            user_id, name = cur.fetchone()
            cur.execute("SELECT count(order_id) FROM orders WHERE user_id=?", (user_id,))
            noOfOrderss = cur.fetchone()[0]
            cur.execute("SELECT count(kart_id) FROM kart WHERE user_id=?", (user_id,))
            noOfItems = cur.fetchone()[0]
            cur.execute("SELECT count(favorite_id) FROM favorite WHERE user_id=?", (user_id,))
            noOfFavorite = cur.fetchone()[0]
    conn.close()
    return (loggedIn, name, noOfOrderss, noOfItems, noOfFavorite)


def parse(data):
    ans = []
    i = 0
    while i < len(data):
        curr = []
        for j in range(7):
            if i >= len(data):
                break
            curr.append(data[i])
            i += 1
        ans.append(curr)
    return ans


@app.route("/ebuy")
def ebuy():
    loggedIn, name, noOfOrderss, noOfItems, noOfFavorite = getLoginDetails()
    with sqlite3.connect('data_management.db') as conn:
        cur = conn.cursor()
        cur.execute('SELECT product_id, name, price, description, image, stock FROM products')
        itemData1 = cur.fetchall()
        cur.execute('SELECT categoryId, name FROM categories')
        categoryData = cur.fetchall()
    itemData = parse(itemData1)
    return render_template('ebuy.html', itemData=itemData, loggedIn=loggedIn, name=name, noOfItems=noOfItems,
                           noOfOrderss=noOfOrderss, noOfFavorite=noOfFavorite, categoryData=categoryData)


@app.route("/loginForm")
def loginForm():
    if 'email' in session:
        return redirect(url_for('ebuy'))
    else:
        return render_template('userlogin.html', error='')


@app.route("/userlogin", methods=['POST', 'GET'])
def userlogin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if is_valid(email, password):
            session['email'] = email
            return redirect(url_for('ebuy'))
        else:
            error = 'Invalid UserId / Password'
            return render_template('userlogin.html', error=error)


def is_valid(email, password):
    con = sqlite3.connect('data_management.db')
    cur = con.cursor()
    cur.execute('SELECT email, password FROM user_info')
    data = cur.fetchall()
    for row in data:
        if row[0] == email and row[1] == hashlib.md5(password.encode()).hexdigest():
            return True
    return False


@app.route("/registerationForm")
def registrationForm():
    return render_template("registeration.html")


@app.route("/registeration", methods=['GET', 'POST'])
def registeration():
    if request.method == 'POST':
        # Parse form data
        password = request.form['password']
        email = request.form['email']
        name = request.form['name']
        gender = request.form['gender']
        create_time = time.time()
        user_id = int(str(np.random.randint(1, 100)) + str(time.time())[0:8])
        with sqlite3.connect('data_management.db') as con:
            try:
                cur = con.cursor()
                cur.execute(
                    'INSERT INTO user_info (user_id,password, email, name, gender,create_time) VALUES (?, ?, ?, ?,?,?)',
                    (user_id, hashlib.md5(password.encode()).hexdigest(), email, name, gender, create_time))

                con.commit()

                msg = "Registered Successfully"
            except:
                con.rollback()
                traceback.print_exc()
                msg = "Error occured"
        con.close()
        return render_template("userlogin.html", error=msg)


@app.route("/logout")
def logout():
    session.pop('email', None)
    return redirect(url_for('ebuy'))


@app.route("/account/profile/edit")
def editProfile():
    if 'email' not in session:
        return redirect(url_for('root'))
    loggedIn, name, noOfOrderss, noOfItems, noOfFavorite = getLoginDetails()
    with sqlite3.connect('data_management.db') as conn:
        cur = conn.cursor()
        cur.execute("SELECT user_id, email, name, gender,password FROM user_info WHERE email = ?", (session['email'],))
        profileData = cur.fetchone()
    conn.close()
    return render_template("editProfile.html", profileData=profileData, loggedIn=loggedIn, name=name,
                           noOfItems=noOfItems, noOfOrderss=noOfOrderss, noOfFavorite=noOfFavorite)


@app.route("/updateProfile", methods=["GET", "POST"])
def updateProfile():
    if request.method == 'POST':
        email = request.form['email']
        name = request.form['name']
        gender = request.form['gender']
        password = request.form['password']
        with sqlite3.connect('data_management.db') as con:
            try:
                cur = con.cursor()
                cur.execute('UPDATE user_info SET name = ?,gender=?,password=? WHERE email = ?',
                            (name, gender, password, email))
                con.commit()
                msg = "Saved Successfully"
            except:
                con.rollback()
                msg = "Error occured"
        con.close()
        return redirect(url_for('ebuy'))


@app.route("/addToCart")
def addToCart():
    if 'email' not in session:
        return redirect(url_for('loginForm'))
    else:
        product_id = int(request.args.get('product_id'))
        with sqlite3.connect('data_management.db') as conn:
            cur = conn.cursor()
            cur.execute("SELECT user_id FROM user_info WHERE email = ?", (session['email'],))
            user_id = cur.fetchone()[0]
            kart_id = int(str(np.random.randint(1, 100)) + str(time.time())[0:8])
            try:
                cur.execute("INSERT INTO kart (kart_id,user_id, product_id) VALUES (?, ?,?)",
                            (kart_id, user_id, product_id))
                conn.commit()
                msg = "Added successfully"
            except:
                conn.rollback()
                traceback.print_exc()
                msg = "Error occured"
            cur.execute('SELECT categoryId FROM products WHERE product_id = ?', (product_id,))
            categoryId = cur.fetchone()[0]
            # Trigger and report to the user behavior table- cart type
            create_time = time.time()
            behavior_id = int(str(np.random.randint(1, 100)) + str(time.time())[0:8])
            date = time.strftime('%Y-%m-%d', time.localtime(create_time))
            year = time.localtime(create_time).tm_year
            month = time.localtime(create_time).tm_mon
            day = time.localtime(create_time).tm_mday
            hour = time.localtime(create_time).tm_hour
            cur.execute("INSERT INTO user_behavior (behavior_id,user_id, product_id,categoryId,behavior_type,create_time,date,year,month,day,hour) \
                  VALUES (?, ?,?,?, ?,?,?, ?,?,?, ?)", (
            behavior_id, user_id, product_id, categoryId, 'cart', create_time, date, year, month, day, hour))
        conn.close()
        return redirect(url_for('ebuy'))


@app.route("/cart")
def cart():
    if 'email' not in session:
        return redirect(url_for('loginForm'))
    loggedIn, name, noOfOrderss, noOfItems, noOfFavorite = getLoginDetails()
    email = session['email']
    with sqlite3.connect('data_management.db') as conn:
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM user_info WHERE email = ?", (email,))
        userId = cur.fetchone()[0]
        cur.execute(
            "SELECT products.product_id, products.name, products.price, products.image,products.stock FROM products, kart WHERE products.product_id = kart.product_id AND kart.user_id = ?",
            (userId,))
        products = cur.fetchall()
    totalPrice = 0
    for row in products:
        totalPrice += row[2]
    return render_template("cart.html", products=products, totalPrice=totalPrice, loggedIn=loggedIn, name=name,
                           noOfItems=noOfItems, noOfOrderss=noOfOrderss, noOfFavorite=noOfFavorite)


@app.route("/removeFromCart")
def removeFromCart():
    if 'email' not in session:
        return redirect(url_for('loginForm'))
    email = session['email']
    product_id = int(request.args.get('product_id'))
    with sqlite3.connect('data_management.db') as conn:
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM user_info WHERE email = ?", (email,))
        user_id = cur.fetchone()[0]
        try:
            cur.execute("DELETE FROM kart WHERE user_id = ? AND product_id = ?", (user_id, product_id))
            conn.commit()
            msg = "removed successfully"
        except:
            conn.rollback()
            msg = "error occured"
    conn.close()
    return redirect(url_for('cart'))


@app.route("/displayCategory")
def displayCategory():
    loggedIn, name, noOfOrderss, noOfItems, noOfFavorite = getLoginDetails()
    categoryId = request.args.get("categoryId")
    with sqlite3.connect('data_management.db') as conn:
        cur = conn.cursor()
        cur.execute("SELECT products.product_id, products.name, products.price, products.image, categories.name \
                      FROM products, categories WHERE products.categoryId = categories.categoryId AND categories.categoryId = ?",
                    (categoryId,))
        data = cur.fetchall()
    conn.close()
    categoryName = data[0][4]
    data = parse(data)
    return render_template('displayCategory.html', data=data, loggedIn=loggedIn, name=name, noOfItems=noOfItems,
                           noOfOrderss=noOfOrderss, noOfFavorite=noOfFavorite, categoryName=categoryName)


@app.route("/productDetails")
def productDetails():
    loggedIn, name, noOfOrderss, noOfItems, noOfFavorite = getLoginDetails()
    product_id = request.args.get('product_id')
    with sqlite3.connect('data_management.db') as conn:
        cur = conn.cursor()
        # Get the content of the product details page
        cur.execute(
            'SELECT product_id, name, price, description, image, stock,categoryId FROM products WHERE product_id = ?',
            (product_id,))
        productData = cur.fetchone()
        # Trigger and report to the user behavior table- PV type
        if 'email' in session:
            cur.execute("SELECT user_id FROM user_info WHERE email = ?", (session['email'],))
            user_id = cur.fetchone()[0]
            behavior_type = 'pv'
            categoryId = productData[6]
            create_time = time.time()
            behavior_id = int(str(np.random.randint(1, 100)) + str(time.time())[0:8])
            date = time.strftime('%Y-%m-%d', time.localtime(create_time))
            year = time.localtime(create_time).tm_year
            month = time.localtime(create_time).tm_mon
            day = time.localtime(create_time).tm_mday
            hour = time.localtime(create_time).tm_hour
            cur.execute("INSERT INTO user_behavior (behavior_id,user_id, product_id,categoryId,behavior_type,create_time,date,year,month,day,hour) \
                  VALUES (?, ?,?,?, ?,?,?, ?,?,?, ?)", (
            behavior_id, user_id, product_id, categoryId, behavior_type, create_time, date, year, month, day, hour))
    conn.close()
    return render_template("productDetails.html", data=productData, loggedIn=loggedIn, name=name, noOfItems=noOfItems,
                           noOfOrderss=noOfOrderss, noOfFavorite=noOfFavorite)


@app.route("/addToFavorite")
def addToFavorite():
    if 'email' not in session:
        return redirect(url_for('loginForm'))
    else:
        loggedIn, name, noOfOrderss, noOfItems, noOfFavorite = getLoginDetails()
        product_id = int(request.args.get('product_id'))
        with sqlite3.connect('data_management.db') as conn:
            cur = conn.cursor()
            cur.execute("SELECT user_id FROM user_info WHERE email = ?", (session['email'],))
            user_id = cur.fetchone()[0]
            favorite_id = int(str(np.random.randint(1, 100)) + str(time.time())[0:8])
            try:
                cur.execute("INSERT INTO favorite (favorite_id,user_id, product_id) VALUES (?, ?,?)",
                            (favorite_id, user_id, product_id))
                conn.commit()
                msg = "Added successfully"
            except:
                conn.rollback()
                traceback.print_exc()
                msg = "Error occured"
            cur.execute(
                "SELECT products.product_id, products.name, products.price, products.image,products.stock,products.categoryId FROM products, favorite WHERE products.product_id = favorite.product_id AND favorite.user_id = ?",
                (user_id,))
            products = cur.fetchall()
            # Trigger and report to the user behavior table- FAV type
            create_time = time.time()
            behavior_type = 'fav'
            behavior_id = int(str(np.random.randint(1, 100)) + str(time.time())[0:8])
            date = time.strftime('%Y-%m-%d', time.localtime(create_time))
            year = time.localtime(create_time).tm_year
            month = time.localtime(create_time).tm_mon
            day = time.localtime(create_time).tm_mday
            hour = time.localtime(create_time).tm_hour
            cur.execute("INSERT INTO user_behavior (behavior_id,user_id, product_id,categoryId,behavior_type,create_time,date,year,month,day,hour) \
                  VALUES (?, ?,?,?, ?,?,?, ?,?,?, ?)", (
            behavior_id, user_id, product_id, products[0][5], behavior_type, create_time, date, year, month, day, hour))
        conn.close()
    return render_template("favorite.html", products=products, loggedIn=loggedIn, name=name, noOfItems=noOfItems,
                           noOfOrderss=noOfOrderss, noOfFavorite=noOfFavorite)


@app.route("/favorite")
def favorite():
    if 'email' not in session:
        return redirect(url_for('loginForm'))
    loggedIn, name, noOfOrderss, noOfItems, noOfFavorite = getLoginDetails()
    email = session['email']
    with sqlite3.connect('data_management.db') as conn:
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM user_info WHERE email = ?", (email,))
        userId = cur.fetchone()[0]
        cur.execute(
            "SELECT products.product_id, products.name, products.price, products.image,products.stock FROM products, favorite WHERE products.product_id = favorite.product_id AND favorite.user_id = ?",
            (userId,))
        products = cur.fetchall()
    return render_template("favorite.html", products=products, loggedIn=loggedIn, name=name, noOfItems=noOfItems,
                           noOfOrderss=noOfOrderss, noOfFavorite=noOfFavorite)


@app.route("/removeFromFavorite")
def removeFromFavorite():
    if 'email' not in session:
        return redirect(url_for('loginForm'))
    email = session['email']
    product_id = int(request.args.get('product_id'))
    with sqlite3.connect('data_management.db') as conn:
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM user_info WHERE email = ?", (email,))
        user_id = cur.fetchone()[0]
        try:
            cur.execute("DELETE FROM favorite WHERE user_id = ? AND product_id = ?", (user_id, product_id))
            conn.commit()
            msg = "removed successfully"
        except:
            conn.rollback()
            msg = "error occured"
    conn.close()
    return redirect(url_for('favorite'))


@app.route("/addToOrders")
def addToOrders():
    if 'email' not in session:
        return redirect(url_for('loginForm'))
    else:
        loggedIn, name, noOfOrderss, noOfItems, noOfFavorite = getLoginDetails()
        product_id = int(request.args.get('product_id'))
        with sqlite3.connect('data_management.db') as conn:
            cur = conn.cursor()
            cur.execute("SELECT user_id FROM user_info WHERE email = ?", (session['email'],))
            user_id = cur.fetchone()[0]
            cur.execute("SELECT name, price,categoryId FROM products WHERE product_id = ?", (product_id,))
            products = cur.fetchall()
            product_name = products[0][0]
            product_price = products[0][1]
            categoryId = products[0][2]
            create_time = time.time()
            date = time.strftime('%Y-%m-%d', time.localtime(create_time))
            year = time.localtime(create_time).tm_year
            month = time.localtime(create_time).tm_mon
            day = time.localtime(create_time).tm_mday
            hour = time.localtime(create_time).tm_hour
            order_id = int(str(np.random.randint(1, 100)) + str(time.time())[0:10])
            try:
                cur.execute("INSERT INTO orders (order_id,user_id, product_id,product_name,product_price,create_time,date,year,month,day,hour) \
                VALUES (?, ?,?,?,?,?,?,?,?,?,?)", (
                order_id, user_id, product_id, product_name, product_price, create_time, date, year, month, day, hour))
                conn.commit()
                msg = "Added successfully"
            except:
                conn.rollback()
                traceback.print_exc()
                msg = "Error occured"
            cur.execute("SELECT product_name, product_price, date FROM orders WHERE user_id = ?", (user_id,))
            orders = cur.fetchall()
            # Trigger and report to the user behavior table- BUY type
            behavior_type = 'buy'
            behavior_id = int(str(np.random.randint(1, 100)) + str(time.time())[0:8])
            date = time.strftime('%Y-%m-%d', time.localtime(create_time))
            year = time.localtime(create_time).tm_year
            month = time.localtime(create_time).tm_mon
            day = time.localtime(create_time).tm_mday
            hour = time.localtime(create_time).tm_hour
            cur.execute("INSERT INTO user_behavior (behavior_id,user_id, product_id,categoryId,behavior_type,create_time,date,year,month,day,hour) \
                  VALUES (?, ?,?,?, ?,?,?, ?,?,?, ?)", (
            behavior_id, user_id, product_id, categoryId, behavior_type, create_time, date, year, month, day, hour))
        conn.close()
        return render_template("orders.html", orders=orders, loggedIn=loggedIn, name=name, noOfItems=noOfItems,
                               noOfOrderss=noOfOrderss, noOfFavorite=noOfFavorite)


@app.route("/orders")
def Orders():
    if 'email' not in session:
        return redirect(url_for('loginForm'))
    loggedIn, name, noOfOrderss, noOfItems, noOfFavorite = getLoginDetails()
    email = session['email']
    with sqlite3.connect('data_management.db') as conn:
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM user_info WHERE email = ?", (email,))
        user_id = cur.fetchone()[0]
        cur.execute("SELECT product_name, product_price, date FROM orders WHERE user_id = ?", (user_id,))
        orders = cur.fetchall()
        conn.rollback()
        traceback.print_exc()
    return render_template("orders.html", orders=orders, loggedIn=loggedIn, name=name, noOfItems=noOfItems,
                           noOfOrderss=noOfOrderss, noOfFavorite=noOfFavorite)


@app.route("/add")
def admin():
    with sqlite3.connect('data_management.db') as conn:
        cur = conn.cursor()
        cur.execute("SELECT categoryId, name FROM categories")
        categories = cur.fetchall()
    conn.close()
    return render_template('add.html', categories=categories)


@app.route("/addItem", methods=["GET", "POST"])
def addItem():
    if request.method == "POST":
        name = request.form['name']
        price = float(request.form['price'])
        description = request.form['description']
        stock = int(request.form['stock'])
        categoryId = int(request.form['category'])

        # Uploading image procedure
        image = request.files['image']
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        imagename = filename
        with sqlite3.connect('data_management.db') as conn:
            try:
                cur = conn.cursor()
                cur.execute(
                    '''INSERT INTO products (name, price, description, image, stock, categoryId) VALUES (?, ?, ?, ?, ?, ?)''',
                    (name, price, description, imagename, stock, categoryId))
                conn.commit()
                msg = "added successfully"
            except:
                msg = "error occured"
                conn.rollback()
        conn.close()
        return redirect(url_for('ebuy'))


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def csv_behavior_insert():
    behavior = pd.read_csv('D:/data/e-commerce01.csv')
    with sqlite3.connect('data_management.db') as conn:
        cur = conn.cursor()
        for i, r in behavior.iterrows():
            try:
                cur.execute("INSERT INTO user_behavior \
                           (behavior_id,user_id,product_id,categoryId,behavior_type,create_time,date,year,month,day,hour) \
                            VALUES (?, ?,?,?, ?,?,?,?, ?,?,?)", \
                            (r[0], r[1], r[2], r[3], r[4], r[5], time.strftime("%Y-%m-%d", time.localtime(r[5])),
                             time.localtime(r[5]).tm_year, time.localtime(r[5]).tm_mon, time.localtime(r[5]).tm_mday,
                             time.localtime(r[5]).tm_hour))
                conn.commit()
            except:
                conn.rollback()
                traceback.print_exc()
                msg = "Error occured"
    conn.close()


def csv_order_insert():
    behavior = pd.read_csv('D:/data/e-commerce02.csv')
    with sqlite3.connect('data_management.db') as conn:
        cur = conn.cursor()
        for i, r in behavior.iterrows():
            try:
                cur.execute("INSERT INTO orders \
                           (order_id,user_id,product_id,product_price,product_name,create_time,date,year,month,day,hour) \
                            VALUES (?, ?,?,?, ?,?,?,?, ?,?,?)", \
                            (r[0], r[1], r[2], r[3], r[4], r[5], time.strftime("%Y-%m-%d", time.localtime(r[5])),
                             time.localtime(r[5]).tm_year, time.localtime(r[5]).tm_mon, time.localtime(r[5]).tm_mday,
                             time.localtime(r[5]).tm_hour))
                conn.commit()
            except:
                conn.rollback()
                traceback.print_exc()
                msg = "Error occured"
    conn.close()


######## draw user activity plot
def user_activity_daily():
    with sqlite3.connect('data_management.db') as conn:
        cur = conn.cursor()
        cur.execute("SELECT date,COUNT(DISTINCT user_id) as UV, COUNT(user_id) as PV \
                      FROM user_behavior GROUP BY date")
        date = []
        uv = []
        pv = []
        for r in cur.fetchall():
            date.append(r[0])
            uv.append(r[1])
            pv.append(r[2])
        fig, ax = plt.subplots(1, 2, figsize=(16, 4))
        fig.suptitle('Daily Activity Chart', fontsize=14)
        fig.autofmt_xdate(rotation=45)
        ax[0].bar(x=date, height=uv, label='UV', color='#f9713c')
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('Number')
        ax[0].legend()

        ax[1].bar(x=date, height=pv, label='PV', color='#f9713c')
        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('Number')
        ax[1].legend()

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
    conn.close()
    return imd


def user_activity_hour():
    with sqlite3.connect('data_management.db') as conn:
        cur = conn.cursor()
        cur.execute("SELECT hour,COUNT(DISTINCT user_id) as UV, COUNT(user_id) as PV \
                     FROM user_behavior  GROUP BY hour")
        hour = []
        uv = []
        pv = []
        for r in cur.fetchall():
            hour.append(r[0])
            uv.append(r[1])
            pv.append(r[2])
        fig, ax = plt.subplots(1, 2, figsize=(16, 4))
        fig.suptitle('Hour Activity Chart', fontsize=14)
        ax[0].bar(x=hour, height=uv, label='UV', color='#b3e4a1')
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('Number')
        ax[0].legend()

        ax[1].bar(x=hour, height=pv, label='PV', color='#b3e4a1')
        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('Number')
        ax[1].legend()
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
    conn.close()
    return imd


def average_pv():
    with sqlite3.connect('data_management.db') as conn:
        cur = conn.cursor()
        cur.execute("SELECT date,COUNT(DISTINCT user_id) as UV, COUNT(user_id) as PV, \
                     COUNT(user_id)/COUNT(DISTINCT user_id) AS meanPVofUV  FROM user_behavior \
                      GROUP BY date")
        date = []
        meanPVofUV_day = []
        for r in cur.fetchall():
            date.append(r[0])
            meanPVofUV_day.append(r[3])
        cur.execute("SELECT hour,COUNT(DISTINCT user_id) as UV, COUNT(user_id) as PV, \
                     COUNT(user_id)/COUNT(DISTINCT user_id) AS meanPVofUV  FROM user_behavior \
                      GROUP BY hour")
        hour = []
        meanPVofUV_hour = []
        for r in cur.fetchall():
            hour.append(r[0])
            meanPVofUV_hour.append(r[3])
        fig, ax = plt.subplots(1, 2, figsize=(16, 4))
        fig.suptitle('PV per capita', fontsize=14)
        fig.autofmt_xdate(rotation=45)
        ax[0].bar(date, meanPVofUV_day, color='#5CACEE')
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('PV')
        ax[0].hlines(np.mean(meanPVofUV_day), date[0], date[-1], linestyle='dashed', lw=0.5)

        ax[1].bar(hour, meanPVofUV_hour, color='#5CACEE')
        ax[1].set_xlabel('hour')
        ax[1].set_ylabel('PV')
        ax[1].hlines(np.mean(meanPVofUV_hour), hour[0], hour[-1], linestyle='dashed', lw=0.5)
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
    conn.close()
    return imd


###############################Funnel Model for conversion rate【Browse--Purchase&Favorite--Purchase】####################################
# the action Funnel Model
def conversion_rate_action():
    with sqlite3.connect('data_management.db') as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(behavior_type) FROM user_behavior WHERE behavior_type='pv'")
        pv_sum = cur.fetchall()
        cur.execute("SELECT COUNT(behavior_type) FROM user_behavior WHERE behavior_type='cart' or behavior_type='fav'")
        cf_sum = cur.fetchall()
        cur.execute("SELECT COUNT(behavior_type) FROM user_behavior WHERE behavior_type='buy'")
        buy_sum = cur.fetchall()
        convert_rate_action = [100, cf_sum[0][0] / pv_sum[0][0] * 100, buy_sum[0][0] / cf_sum[0][0] * 100]
        x_data = ["Website visit", "Add to cart & Add to favorite", "Buy product"]

        data = [[x_data[i], convert_rate_action[i]] for i in range(len(x_data))]

        (
            Funnel()
            .add(
                series_name="",
                data_pair=data,
                gap=2,
                tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{a} <br/>{b} : {c}%"),
                label_opts=opts.LabelOpts(is_show=True, position="inside"),
                itemstyle_opts=opts.ItemStyleOpts(border_color="#fff", border_width=1),
            )
            .set_global_opts(title_opts=opts.TitleOpts(title="Consumer action funnel",
                                                       subtitle="Browse --> Purchase&Favorite --> Purchase"))
            .render("templates/funnel_action.html")
        )
    conn.close()


def conversion_rate_individual():
    with sqlite3.connect('data_management.db') as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(DISTINCT(user_id)) FROM user_behavior WHERE behavior_type='pv'")
        user_pv = cur.fetchall()
        cur.execute(
            "SELECT COUNT(DISTINCT(user_id)) FROM user_behavior WHERE behavior_type='cart' or behavior_type='fav'")
        user_cf = cur.fetchall()
        cur.execute("SELECT COUNT(DISTINCT(user_id)) FROM user_behavior WHERE behavior_type='buy'")
        user_buy = cur.fetchall()
        convert_rate_individual = [100, user_cf[0][0] / user_pv[0][0] * 100, user_buy[0][0] / user_cf[0][0] * 100]
        x_data = ["Website visit", "Add to cart & Add to favorite", "Buy product"]

        data = [[x_data[i], convert_rate_individual[i]] for i in range(len(x_data))]

        (
            Funnel()
            .add(
                series_name="",
                data_pair=data,
                gap=2,
                tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{a} <br/>{b} : {c}%"),
                label_opts=opts.LabelOpts(is_show=True, position="inside"),
                itemstyle_opts=opts.ItemStyleOpts(border_color="#fff", border_width=1),
            )
            .set_global_opts(title_opts=opts.TitleOpts(title="Individual funnel plot",
                                                       subtitle="Browse --> Purchase&Favorite --> Purchase"))
            .render("templates/funnel_individual.html")
        )
    conn.close()


############################################# User order plot #############################################
def user_order():
    with sqlite3.connect('data_management.db') as conn:
        cur = conn.cursor()
        cur.execute("SELECT date,COUNT(DISTINCT user_id), COUNT(product_id) ,SUM(product_price) \
                      FROM orders GROUP BY date")
        date = []
        noOfBuyer = []
        noOfOrder = []
        GMV = []
        for r in cur.fetchall():
            date.append(r[0])
            noOfBuyer.append(r[1])
            noOfOrder.append(r[2])
            GMV.append(r[3])
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle('User order data graph', fontsize=14)
        fig.autofmt_xdate(rotation=45)
        ax[0].bar(date, noOfBuyer, label='Buyer', color='#f9713c')
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('Number')
        ax[0].legend()
        ax[1].bar(date, noOfOrder, label='Orders', color='#b3e4a1')
        ax[1].set_xlabel('Date')
        ax[1].legend()
        ax[2].bar(date, GMV, label='GMV', color='#5CACEE')
        ax[2].set_xlabel('Date')
        ax[2].legend()
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
    conn.close()
    return imd


def hot_product():
    with sqlite3.connect('data_management.db') as conn:
        cur = conn.cursor()
        cur.execute("SELECT product_name, COUNT(order_id), SUM(product_price) \
                      FROM orders GROUP BY product_id")
        product = []
        sales_volume = []
        sales_money = []
        for r in cur.fetchall():
            product.append(r[0])
            sales_volume.append(r[1])
            sales_money.append(r[2])
        fig, ax = plt.subplots(1, 2, figsize=(16, 4))
        fig.suptitle('Hot product chart', fontsize=14)
        fig.autofmt_xdate(rotation=45)
        ax[0].bar(product, sales_volume, label='Sales Volume', color='red')
        ax[0].set_xlabel('Products')
        ax[0].set_ylabel('Number')
        ax[0].legend()

        ax[1].bar(product, sales_money, label='Sales Money', color='orange')
        ax[1].set_xlabel('Products')
        ax[0].set_ylabel('Sales Money')
        ax[1].legend()
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
    conn.close()
    return imd


########################################## Using RFM and Kmeans to classify buyers ###########################
def RFM_kmeans():
    with sqlite3.connect('data_management.db') as conn:
        cur = conn.cursor()
        cur.execute("SELECT user_id, MIN(18-day),COUNT(order_id),SUM(product_price) FROM orders GROUP BY user_id")
        user_id = []
        recency = []
        frequency = []
        monetary = []
        for x in cur.fetchall():
            user_id.append(x[0])
            recency.append(x[1])
            frequency.append(x[2])
            monetary.append(x[3])
        rfm = pd.DataFrame({'user_id': user_id, 'recency': recency, 'frequency': frequency, 'monetary': monetary})
        # z-score stadardize data
        standardizer = StandardScaler()
        std_data = pd.DataFrame(standardizer.fit_transform(rfm.iloc[:, [1, 2, 3]]))
        mean_X, std_X = standardizer.mean_, standardizer.scale_
        # kmeans group buyer
        my_kmeans = KMeans(n_clusters=3, random_state=10)
        my_kmeans.fit(std_data)
        clusters = my_kmeans.predict(std_data)
        centroids = my_kmeans.cluster_centers_
        # group results
        rfm['group'] = clusters
        # Cluster center coordinate restoration
        centroids_new = centroids * std_X + mean_X
    return rfm, centroids_new


def radar_group():
    rfm, centroids_new = RFM_kmeans()
    group1 = [list(centroids_new[0])]
    group2 = [list(centroids_new[1])]
    group3 = [list(centroids_new[2])]
    radar = (
        Radar()
        .add_schema(
            schema=[
                opts.RadarIndicatorItem(name="recency", max_=max(rfm['recency'])),
                opts.RadarIndicatorItem(name="frequency", max_=max(rfm['frequency'])),
                opts.RadarIndicatorItem(name="monetary", max_=max(rfm['monetary']))]

        )
        .add("group1", group1, color="#f9713c")
        .add("group2", group2, color="#b3e4a1")
        .add("group3", group3, color="#5CACEE")
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            legend_opts=opts.LegendOpts(),
            title_opts=opts.TitleOpts(title="Radar-Buyer Group"),
        ).render("templates/radar_buyer_mode.html")
    )
    return radar


def visualization_group():
    rfm, centroids_new = RFM_kmeans()
    plt.figure(figsize=(6, 4))
    plt.ylim(0, 30)
    plt.bar(range(3), rfm.groupby('group')['group'].count())
    plt.xticks(range(3), ['group1', 'group2', 'group3'])
    x = [0, 1, 2]
    y = rfm.groupby('group')['group'].count() + 2
    text = rfm.groupby('group')['group'].count()
    i = 0
    for a, b in zip(x, y):
        plt.text(a, b, text[i])
        i += 1
    plt.show()


@app.route('/user/activity')
def activity():
    imd1 = user_activity_daily()
    imd2 = user_activity_hour()
    imd3 = average_pv()
    return render_template('/userVisual.html', img=imd1, img2=imd2, img3=imd3)


@app.route('/user/order')
def order():
    imd1 = user_order()
    imd2 = hot_product()
    imd3 = None
    return render_template('/userVisual.html', img=imd1, img2=imd2, img3=imd3)


@app.route('/user/radar')
def radar():
    rfm, centroids_new = RFM_kmeans()
    res = []
    for i in range(len(rfm)):
        res.append(tuple(rfm.iloc[i]))
    return render_template('/radar_buyer_mode.html', res=res)


@app.route('/user/funnel01')
def funnel01():
    #    conversion_rate_action()
    return render_template('/funnel_action.html')


@app.route('/user/funnel02')
def funnel02():
    #    conversion_rate_individual()
    return render_template('/funnel_individual.html')


if __name__ == '__main__':
    # sql_init()
    # app.run('localhost', 9002)
    run_simple('localhost', 9001, app)
