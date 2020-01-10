package com.example.yubix.encogg;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationReLU;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


    }

    public void submit(View view) {
        EditText input11 = findViewById(R.id.input11);
        EditText input12 = findViewById(R.id.input12);
        EditText input21 = findViewById(R.id.input21);
        EditText input22 = findViewById(R.id.input22);
        EditText input31 = findViewById(R.id.input31);
        EditText input32 = findViewById(R.id.input32);
        EditText input41 = findViewById(R.id.input41);
        EditText input42 = findViewById(R.id.input42);
        double XOR_INPUT[][] = { {Double.parseDouble(input11.getText().toString()), Double.parseDouble(input12.getText().toString())},
                {Double.parseDouble(input21.getText().toString()), Double.parseDouble(input22.getText().toString())},
                {Double.parseDouble(input31.getText().toString()), Double.parseDouble(input32.getText().toString())},
                {Double.parseDouble(input41.getText().toString()), Double.parseDouble(input42.getText().toString())}, };

        String output = runModel(XOR_INPUT);
        TextView text = (TextView) findViewById(R.id.text);
        text.setText(output);
    }

    //public static double XOR_INPUT[][] = { { 0.0, 0.0 }, { 1.0, 0.0 },
           // { 0.0, 1.0 }, { 1.0, 1.0 } };

    public static double XOR_IDEAL[][] = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } };

    public String runModel(double[][] XOR_INPUT) {

        // create a neural network, without using a factory
        String result = "";
        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null,true,2));
        network.addLayer(new BasicLayer(new ActivationReLU(),true,5));
        network.addLayer(new BasicLayer(new ActivationSigmoid(),false,1));
        network.getStructure().finalizeStructure();
        network.reset();

        // create training data
        MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);

        // train the neural network
        final ResilientPropagation train = new ResilientPropagation(network, trainingSet);

        int epoch = 1;

        do {
            train.iteration();
            result += "\nEpoch #" + epoch + " Error:" + train.getError();
            epoch++;
        } while(train.getError() > 0.01);
        train.finishTraining();

        // test the neural network
        result += "\nNeural Network Results:";
        for(MLDataPair pair: trainingSet ) {
            final MLData output = network.compute(pair.getInput());
            result += "\n" + pair.getInput().getData(0) + "," + pair.getInput().getData(1)
                    + ", actual=" + output.getData(0) + ",ideal=" + pair.getIdeal().getData(0);
        }

        Encog.getInstance().shutdown();

        return result;
    }
}
