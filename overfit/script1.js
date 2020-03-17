import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { getData } from './data.js'

window.onload = async () => {
    const data = getData(200, 2);  // 参数为样本数量和方差
    
    tfvis.render.scatterplot(
        { name: '训练数据' },
        {
            values: [
                data.filter(p => p.label === 1),
                data.filter(p => p.label === 0)
            ]
        }
    );

};