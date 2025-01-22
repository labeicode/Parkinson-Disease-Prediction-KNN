def knn_analysis(file_path):
    import pandas
    import sklearn
    from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    import io
    import base64
    import json

    # 读取数据集
    dataset = pandas.read_csv(file_path)

    # 将数据集转换为数组以便处理
    array = dataset.values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(array)
    
    # X 存储特征值
    X = scaled[:,0:22]
    # Y 存储标签
    Y = scaled[:,22]
    validation_size = 0.25
    seed = 7
    
    # 将数据集拆分为训练集和验证集
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
    print(X_train)
    
    # 10折交叉验证
    num_folds = 10
    num_instances = len(X_train)
    scoring = 'accuracy'
    
    clf = KNeighborsClassifier()
    kfold = sklearn.model_selection.KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(clf, X_train, Y_train, cv=kfold, scoring=scoring)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_validation)
    
    # 配置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 准确率可视化 - 饼图
    plt.figure(figsize=(8, 4))
    accuracy = accuracy_score(Y_validation, predictions) * 100
    plt.pie([accuracy, 100-accuracy], labels=['正确', '错误'], 
            autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    plt.title('KNN模型准确率')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_accuracy_pie = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # Matthews相关系数可视化 - 柱状图
    plt.figure(figsize=(8, 4))
    mcc = matthews_corrcoef(Y_validation, predictions)
    plt.bar(['Matthews相关系数'], [mcc], color='skyblue')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.title('KNN模型Matthews相关系数')
    plt.ylabel('相关系数值')
    plt.ylim(-1, 1)
    plt.grid(True, axis='y', alpha=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_mcc_bar = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # 混淆矩阵可视化 - 热图
    cm = confusion_matrix(Y_validation, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('KNN模型混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_confusion_matrix = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # 分类报告可视化 - 热图
    report = classification_report(Y_validation, predictions, output_dict=True)
    report_df = pandas.DataFrame(report).transpose()
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='YlOrRd')
    plt.title('KNN模型分类报告')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_classification_report = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # 生成包含base64图片的JSON
    output = [
        {'title': '准确率饼图', 'base64': img_accuracy_pie},
        {'title': 'Matthews相关系数柱状图', 'base64': img_mcc_bar},
        {'title': '混淆矩阵热图', 'base64': img_confusion_matrix},
        {'title': '分类报告热图', 'base64': img_classification_report}
    ]
    
    return json.dumps(output, ensure_ascii=False)


