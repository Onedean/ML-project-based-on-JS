import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { MnistData } from './data.js'

window.onload = async () => {
    const data = new MnistData();  // 新建MnistData实例
    await data.load();  // 异步load加载图片和二进制文件方法

    const examples = data.nextTestBatch(20);  // 加载一些验证集数据
    // console.log(examples);

    const surface = tfvis.visor().surface({ name: '输入示例' });  // 新建visor实例

    for (let i = 0; i < 20; i += 1) {  // 分别提取20个图片作为每一项tensor
        const imageTensor = tf.tidy(() => {  // 清除webgl的内存，防止内存泄漏
            return examples.xs.slice([i, 0], [1, 784]).reshape([28, 28, 1]);  //切割1个图片784（28*28）像素并重塑形状
        });

        const canvas = document.createElement('canvas');  // 创建一个canvas对象
        canvas.width = 28;  // 设置宽高
        canvas.height = 28;
        canvas.style = 'margin:4px';  // 设置边距
        await tf.browser.toPixels(imageTensor, canvas);  // 将tensor像素值渲染绘制到canvas上
        surface.drawArea.appendChild(canvas);  // 添加到浏览器上

    };

    const model = tf.sequential(); // 创建神经网络
    // 第一轮提取（横、竖）
    model.add(tf.layers.conv2d({  // 添加卷积层（图片为2维）
        inputShape: [28, 28, 1],  // 设置28*28像素形状
        kernelSize: 5,  // 设置卷积核大小
        filters: 8,  // 设置过滤次数（超参数）
        strides: 1,  // 设置卷积核移动步长
        activation: 'relu',  // 设置激活函数relu
        kernelInitializer: 'varianceScaling'  // 设置卷积层初始化方法
    }));

    model.add(tf.layers.maxPool2d({  // 添加池化层
        poolSize: [2, 2],  // 设置池化尺寸
        strides: [2, 2]  // 设置移动步数
    }));

    // 第二轮提取（直角、横线等）
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,  // 提取更加复杂的特征需增大
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    model.add(tf.layers.maxPool2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));

    model.add(tf.layers.flatten());  // 将高维数据摊平放到最一层dense层

    model.add(tf.layers.dense({ // 添加全连接层，进行分类
        units: 10,
        activation: 'softmax',
        kernelInitializer: 'varianceScaling'
    }));

    model.compile({
        loss: 'categoricalCrossentropy',  // 设置损失函数为交叉熵
        optimizer: tf.train.adam(),  // 设置优化器
        metrics: ['accuracy']  // 设置度量单位
    });

    const [trainXs, trainYs] = tf.tidy(() => {  // 准备训练集
        const d = data.nextTrainBatch(2000);  // 拿到1000个训练集数据
        return [
            d.xs.reshape([2000, 28, 28, 1]),  // 注意转换为tensor28*28像素形状
            d.labels
        ];
    });

    const [testXs, testYs] = tf.tidy(() => {  // 准备验证集
        const d = data.nextTestBatch(200);
        return [
            d.xs.reshape([200, 28, 28, 1]),
            d.labels
        ];
    });

    await model.fit(trainXs, trainYs, {  // 进行拟合训练
        validationData: [testXs, testYs],
        epochs: 50,
        callbacks: tfvis.show.fitCallbacks(  // 可视化训练过程
            { name: '训练效果' },
            ['loss', 'val_loss', 'acc', 'val_acc'],
            { callbacks: ['onEpochEnd'] }
        )
    });

    const canvas = document.querySelector('canvas');

    canvas.addEventListener('mousemove', (e) => {
        if (e.buttons === 1) {  // 若用户按着左键
            const ctx = canvas.getContext('2d');  // 获取canvas的上下文
            ctx.fillStyle = 'rgb(255,255,255)';  // 填充矩形为白底
            ctx.fillRect(e.offsetX, e.offsetY, 25, 25);  // 画线条
        }
    });

    window.clear = () => {
        const ctx = canvas.getContext('2d');  // 获取canvas的上下文
        ctx.fillStyle = 'rgb(0,0,0)';  // 填充矩形为黑底
        ctx.fillRect(0, 0, 300, 300);  // 从左上角开始画一个300*300矩形
    };

    clear();

    window.predict = () => {
        const input = tf.tidy(() => {
            return tf.image.resizeBilinear(  // 重塑画板canvas300*300形状为28*28
                tf.browser.fromPixels(canvas),  // 获取画板
                [28, 28],
                true
            ).slice([0, 0, 0], [28, 28, 1])  // 转化为黑白图片，即删除另外两个通道
                .toFloat()
                .div(255)  // 数据归一化
                .reshape([1, 28, 28, 1]);  // 重塑形状
        });
        const pred = model.predict(input).argMax(1);  // 预测
        alert(`预测结果为 ${pred.dataSync()[0]}`);
    };

}