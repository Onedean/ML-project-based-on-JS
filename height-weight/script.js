import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

window.onload = async () => {
    // 准备身高体重训练数据
    const heights = [150, 160, 170];
    const weights = [40, 50, 60];

    // 使用tfvis进行可视化
    tfvis.render.scatterplot(
        { name: '身高体重训练数据' },
        { values: heights.map((x, i) => ({ x, y: weights[i] })) },
        { xAxisDomain: [140, 180], yAxisDomain: [30, 70] }
    );

    // 使用tensorflow.js的api进行归一化
    const inputs = tf.tensor(heights).sub(150).div(20);  // 该归一化方法：减去基准150后除以间距20
    const labels = tf.tensor(weights).sub(40).div(20);

    //定义模型结构
    const model = tf.sequential();  // 初始化一个连续的神经网络模型
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));  // 为神经网络模型添加全连接层，其中神经元个数位1，tensor形状为1维
    model.compile({ loss: tf.losses.meanSquaredError, optimizer: tf.train.sgd(0.1) });  // 设置损失函数（采用均方误差MSE），设置优化器（学习速率超参数为0.1）

    // 训练模型并可视化操作过程
    await model.fit(inputs, labels, {  // 拟合
        batchSize: 3,  // 小批量处理超参数,此处数据为3个,默认为32
        epochs: 200,  // 处理次数
        callbacks: tfvis.show.fitCallbacks(  // 可视化
            { name: '训练过程' },
            ['loss']  // 损失函数可视化
        )
    })

    // 进行预测
    const output = model.predict(tf.tensor([180]).sub(150).div(20));  // 将待测数据转化为tensor，使用训练好的模型预测
    alert(`如果 身高 为 180, 那么 体重 为 ${output.mul(20).add(40).dataSync()[0]}`);  // 将输出的tensor转化为普通数据显示，注意反归一化
}