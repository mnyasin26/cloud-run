require('dotenv').config();
const Hapi = require('@hapi/hapi');
const tf = require('@tensorflow/tfjs-node');
const { Firestore } = require('@google-cloud/firestore');
const crypto = require('crypto');
const path = require('path');
const fs = require('fs');
const { Storage } = require('@google-cloud/storage');

const storage = new Storage();

class InputError extends Error {
    constructor(message) {
        super(message);
        this.name = 'InputError';
        this.statusCode = 400;
    }
}

const BUCKET_NAME = 'model_mlgc_yasin';
const MODEL_PATH = 'model.json';

const downloadFile = async (bucketName, srcFilename, destFilename) => {
    const options = {
        destination: destFilename,
    };
    await storage.bucket(bucketName).file(srcFilename).download(options);
    console.log(`Downloaded ${srcFilename} to ${destFilename}`);
};


async function loadModel() {
    const modelDir = path.join(__dirname, 'model');
    if (!fs.existsSync(modelDir)) {
        fs.mkdirSync(modelDir);
    }

    const modelJsonPath = path.join(modelDir, 'model.json');
    await downloadFile(BUCKET_NAME, MODEL_PATH, modelJsonPath);

    const modelWeights = [
        'group1-shard1of4.bin',
        'group1-shard2of4.bin',
        'group1-shard3of4.bin',
        'group1-shard4of4.bin'
    ];

    for (const weight of modelWeights) {
        await downloadFile(BUCKET_NAME, weight, path.join(modelDir, weight));
    }

    model = await tf.loadGraphModel(`file://${modelJsonPath}`);
    return model;
}


async function storeData(id, data) {
    const db = new Firestore();
    const predictCollection = db.collection('predictions');
    return predictCollection.doc(id).set(data);
}


async function predictClassification(model, image) {
    try {
        const tensor = tf.node
            .decodeJpeg(image)
            .resizeNearestNeighbor([224, 224])
            .expandDims()
            .toFloat();
        const prediction = model.predict(tensor);
        const score = await prediction.data();
        const confidenceScore = Math.max(...score) * 100;
        const label = confidenceScore <= 50 ? 'Non-cancer' : 'Cancer';
        let suggestion;
        if (label === 'Cancer') {
            suggestion = "Segera periksa ke dokter!";
        } else {
            suggestion = "Penyakit kanker tidak terdeteksi.";
        }
        return { label, suggestion };
    } catch (error) {
        throw new InputError('Terjadi kesalahan dalam melakukan prediksi');
    }
}

async function predict(request, h) {
    const { image } = request.payload;
    const { model } = request.server.app;
    const { label, suggestion } = await predictClassification(model, image);
    const id = crypto.randomUUID();
    const createdAt = new Date().toISOString();
    const data = {
        id: id,
        result: label,
        suggestion: suggestion,
        createdAt: createdAt
    };
    await storeData(id, data);
    const response = h.response({
        status: 'success',
        message: 'Model is predicted successfully',
        data
    });
    response.code(201);
    return response;
}

async function postPredictHistoriesHandler(request, h) {
    const allData = await getAllData();
    const formatAllData = [];
    allData.forEach(doc => {
        const data = doc.data();
        formatAllData.push({
            id: doc.id,
            history: {
                result: data.result,
                createdAt: data.createdAt,
                suggestion: data.suggestion,
                id: doc.id
            }
        });
    });
    const response = h.response({
        status: 'success',
        data: formatAllData
    });
    response.code(200);
    return response;
}

const routes = [
    {
        path: '/predict',
        method: 'POST',
        handler: predict,
        options: {
            payload: {
                allow: 'multipart/form-data',
                multipart: true,
                maxBytes: 1000000
            }
        }
    },

];

const tls = {
    key: fs.readFileSync(path.join(__dirname, 'cert', 'key.pem')),
    cert: fs.readFileSync(path.join(__dirname, 'cert', 'cert.pem'))
};

(async () => {
    const server = Hapi.server({
        port: process.env.PORT || 3000,
        host: '0.0.0.0',
        tls: tls,
        routes: {
            cors: {
                origin: ['*'],
            },
        },
    });

    const model = await loadModel();
    server.app.model = model;
    server.route(routes);

    server.ext('onPreResponse', function (request, h) {
        const response = request.response;
        if (response instanceof InputError) {
            const newResponse = h.response({
                status: 'fail',
                message: `${response.message}`
            });
            newResponse.code(response.statusCode);
            return newResponse;
        }
        if (response.isBoom) {
            const newResponse = h.response({
                status: 'fail',
                message: response.message
            });
            newResponse.code(response.output.statusCode);
            return newResponse;
        }
        return h.continue;
    });

    await server.start();
    console.log(`Server start at: ${server.info.uri}`);
})();
