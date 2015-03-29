package org.wso2.ml.rbm;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.fetchers.IrisDataFetcher;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.PretrainLayerFactory;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.LoggerFactory;
import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

/**
 * Created by malintha on 3/28/15.
 */
public class InitRBM {

    private static org.slf4j.Logger log = LoggerFactory.getLogger(InitRBM.class);

        public static void main(String[] args) throws IOException {

            System.out.println("Basic classification using RBM\nEnter the dataset\n1.MNIST\n2.IRIS Flower\n");
            InputStreamReader is = new InputStreamReader(System.in);
            BufferedReader br = new BufferedReader(is);
            int datasetNum = Integer.parseInt(br.readLine());
            DataSet ds;
            BaseDataFetcher bdf;
            if(datasetNum==1) {
                bdf = new MnistDataFetcher(true);
            }
            else {
                bdf = new IrisDataFetcher();
            }

            System.out.println("Enter num of Examples to fetch\n");
            int numExamples = Integer.parseInt(br.readLine());

            bdf.fetch(numExamples);
            ds = bdf.next();
            DataSet resultSet = ds.get(100);
            INDArray lbls = resultSet.getLabels();
            for(int i=0;i<lbls.length();i++){
                System.out.println(lbls.getDouble(i));
            }
            System.out.println(resultSet.getLabels());

            IrisDataFetcher fetcher = new IrisDataFetcher();
            fetcher.fetch(150);
            DataSet d = fetcher.next();
            System.out.println(d.numExamples());


        }

}
