package ml.utils;

import ml.classifiers.DecisionTreeClassifier;
import ml.classifiers.DecisionTreeClassifierPruning;
import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;

public class Experimenter {

    public static void main(String[] args){
     DataSet dataSet = new DataSet("/Users/noorakhter/ml-final/ml/utils/2017_Financial_Data_Filled.csv", 0); 
		DataSet dataset2 = new DataSet("/Users/noorakhter/ml-final/ml/utils/2018_Financial_Data_Filled.csv", 0);
		int k = 20; 
		CrossValidationSet cvs = new CrossValidationSet(dataSet, k);
		double totalAccuracy = 0;
	
        // accuracy calculations 
		for (int j = 0; j < k; j++) {
			DataSetSplit split = cvs.getValidationSet(j);
			DecisionTreeClassifierPruning classifier = new DecisionTreeClassifierPruning();
			classifier.train(split.getTrain());
	
			int correct = 0;
			for (Example example : split.getTest().getData()) {
				double predicted = classifier.classify(example);
				if (predicted == example.getLabel()) {
					correct++;
				}
			}
	
			double accuracy = (double) correct / split.getTest().getData().size();
			totalAccuracy += accuracy;
			System.out.println("Accuracy for fold " + (k+1) + ": " + accuracy + " (with pruning)");

            System.out.println("Size of tree: " + classifier.calculateDepth());
		}
	
		double averageAccuracy = totalAccuracy / k;
		System.out.println("Average Accuracy: " + averageAccuracy);

		double totalAccuracy2 = 0;
		for (int i = 0; i < k; i++) {
			DataSetSplit split = cvs.getValidationSet(i);
			DecisionTreeClassifier classifier2 = new DecisionTreeClassifier();
			classifier2.train(split.getTrain());
	
			int correct = 0;
			for (Example example : split.getTest().getData()) {
				double predicted = classifier2.classify(example);
				if (predicted == example.getLabel()) {
					correct++;
				}
			}
	
			double accuracy = (double) correct / split.getTest().getData().size();
			totalAccuracy2 += accuracy;
			System.out.println("Accuracy for fold " + (i+1) + ": " + accuracy + " (without pruning)");
		}
	
		double averageAccuracy2 = totalAccuracy2 / k;
		System.out.println("Average Accuracy: " + averageAccuracy2);

		DecisionTreeClassifierPruning comparison = new DecisionTreeClassifierPruning();
		comparison.train(dataSet);
		double correct2 = 0;
		for (Example examples : dataset2.getData()){
			double predicted = comparison.classify(examples);
				if (predicted == examples.getLabel()) {
					correct2++;
				}
		}
		double accuracy2 = correct2/dataset2.getData().size();
		System.out.println(accuracy2);

        // size of tree
	}
}
