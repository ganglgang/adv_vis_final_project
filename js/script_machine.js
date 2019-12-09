

function init(){
    var data_size = document.getElementById('myRange').value;
    var data_x = $("#data_x :selected").val();
    var data_y = $("#data_y :selected").val();
    var learning_rate = document.getElementById('Learning_Rate').value;
    var batch_size = document.getElementById('Batch_Size').value;
    var epoch = document.getElementById('Epoch').value;
    var loss_function = $("#Loss_Function :selected").val();
    var optimizer_ = $("#Optimizer :selected").val();
    var metrics_ = $("#Metrics :selected").val();
    var regula = $("#regularizers :selected").val();
    var prediction_set =$("input[name=predic]:checked").val();
    run(data_size,data_x,data_y,learning_rate,batch_size,epoch,prediction_set,loss_function,
        optimizer_,metrics_,regula);
}



    /**
     * Get the car data reduced to just the variables we are interested
     * and cleaned of missing data.
     */
    async function getData(size,x,y) {

        // get the selected size data

        var data_set = []
        const DataReq = await fetch('data_correlation.json');
        const Data = await DataReq.json();
        tf.util.shuffle(Data);
        data_new = [];
        data_test = [];
        for (i = 0; i < parseInt(size); i++) {
            data_new.push(Data[i])
            data_test.push(Data[Data.length-i-1]);
        }
        const cleaned_train = data_new.map(element => ({
            x: element[x],
            y: element[y],
        }))
            .filter(element => (element.x != null && element.y != null));
        data_set.push(cleaned_train);
        const cleaned_test = data_test.map(element => ({
            x: element[x],
            y: element[y],
        }))
            .filter(element => (element.x != null && element.y != null));
        data_set.push(cleaned_test);
        return data_set;
    }

    async function run(size,x,y,lr,bs,ep,predictionS,lf,op,me,re) {
        // Load and plot the original input data that we are going to train on.
        var data = await getData(size,x,y);
        var test_data = data[1];
        data = data[0];
        const values = data.map(d => ({
            x: d.x,
            y: d.y,
        }));
        tfvis.render.scatterplot(
            {name: x +' V '+y},
            {values},
            {
                xLabel: x,
                yLabel: y,
                height: 300
            }
        );

        // More code will be added below
        const model = createModel();
        tfvis.show.modelSummary({name: 'Model Summary'}, model);
        const tensorData = convertToTensor(data);
        const {inputs, labels} = tensorData;

        //test data
        if (predictionS == 'train'){
            await trainModel(model, inputs, labels,lr,bs,ep,lf,op,me,re);
            testModel(model, data, tensorData,x,y,'Train Dataset');
        }else{
            const test_tensorData = convertToTensor(test_data);
            await trainModel(model, inputs, labels,lr,bs,ep,lf,op,me,re);
            testModel(model, test_data, test_tensorData,x,y,'Test Dataset');
        }

// Train the model


    }

    function createModel() {
        // Create a sequential model
        const model = tf.sequential();

        // Add a single hidden layer
        model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
        model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}));
        model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}));
        model.add(tf.layers.dense({units: 25, activation: 'sigmoid'}));
        model.add(tf.layers.dense({units: 5, activation: 'sigmoid'}));
        // Add an output layer
        model.add(tf.layers.dense({units: 1, useBias: true}));

        return model;
    }


    /**
     * Convert the input data to tensors that we can use for machine
     * learning. We will also do the important best practices of _shuffling_
     * the data and _normalizing_ the data
     * MPG on the y-axis.
     */
    function convertToTensor(data) {
        // Wrapping these calculations in a tidy will dispose any
        // intermediate tensors.

        return tf.tidy(() => {
            // Step 1. Shuffle the data
            tf.util.shuffle(data);

            // Step 2. Convert data to Tensor
            const inputs = data.map(d => d.x)
            const labels = data.map(d => d.y);

            const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
            const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

            //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
            const inputMax = inputTensor.max();
            const inputMin = inputTensor.min();
            const labelMax = labelTensor.max();
            const labelMin = labelTensor.min();

            const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
            const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

            return {
                inputs: normalizedInputs,
                labels: normalizedLabels,
                // Return the min/max bounds so we can use them later.
                inputMax,
                inputMin,
                labelMax,
                labelMin,
            }
        });
    }

    async function trainModel(model, inputs, labels,l_r,b_s,e_p,lf,op,re) {

        // set hyper parameters
        var learn_rate = parseFloat(l_r);

        //loss function
        if (lf=='meanSquaredError'){
            var loss_function = tf.losses.meanSquaredError;
        }else if(lf=='hingeLoss'){
            var loss_function = tf.losses.hingeLoss
        }else if(lf =='sigmoidCrossEntropy'){
            var loss_function = tf.losses.sigmoidCrossEntropy
        }else if(lf=='softmaxCrossEntropy'){
            var loss_function = tf.losses.softmaxCrossEntropy
        }else if(lf=='cosineDistance'){
            var loss_function = tf.losses.cosineDistance
        }
        //optimizer
        if(op=='adam'){
            var opt = tf.train.adam(learn_rate)
        }else if(op=='sgd'){
            var opt = tf.train.sgd(learn_rate)
        }else if(op=='momentum'){
            var opt = tf.train.momentum(learn_rate)
        }else if (op=='adagrad'){
            var opt = tf.train.adagrad(learn_rate)
        }else if (op=='adadelta'){
            var opt = tf.train.adadelta(learn_rate)
        }else if (op=='adamax'){
            var opt = tf.train.adamax(learn_rate)
        }else if(op=='rmsprop'){
            var opt = tf.train.rmsprop(learn_rate)
        }

        //regulizaer
        if(re=='tf.regularizers.l1l2'){
            var regul = tf.regularizers.l1l2;
        }else if(re=='tf.regularizers.l1'){
            var regul = tf.regularizers.l1;
        }else if(re=='tf.regularizers.l1'){
            var regul = tf.regularizers.l2;
        }


        // Prepare the model for training.
        model.compile({
            optimizer: opt,
            loss: loss_function,
            metrics: ['mse'],
            regularizers:regul

        });
        const batchSize = parseInt(b_s);
        const epochs = parseInt(e_p);

        return await model.fit(inputs, labels, {
            batchSize,
            epochs,
            shuffle: true,
            callbacks: tfvis.show.fitCallbacks(
                {name: 'Training Performance'},
                ['loss', 'mse'],
                {height: 200, callbacks: ['onEpochEnd']}

            )
        });
    }

    function testModel(model, inputData, normalizationData,datax,datay,tag) {
        const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

        // Generate predictions for a uniform range of numbers between 0 and 1;
        // We un-normalize the data by doing the inverse of the min-max scaling
        // that we did earlier.
        const [xs, preds] = tf.tidy(() => {

            const xs = tf.linspace(0, 1, 100);
            const preds = model.predict(xs.reshape([100, 1]));

            const unNormXs = xs
                .mul(inputMax.sub(inputMin))
                .add(inputMin);

            const unNormPreds = preds
                .mul(labelMax.sub(labelMin))
                .add(labelMin);

            // Un-normalize the data
            return [unNormXs.dataSync(), unNormPreds.dataSync()];
        });


        const predictedPoints = Array.from(xs).map((val, i) => {
            return {x: val, y: preds[i]}
        });

        const originalPoints = inputData.map(d => ({
            x: d.x, y: d.y,
        }));


        tfvis.render.scatterplot(
            {name: 'Model Predictions vs '+tag},
            {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
            {
                xLabel: datax,
                yLabel: datay,
                height: 300
            }
        );
    }

