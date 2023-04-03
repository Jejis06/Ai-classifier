#include <iostream>
#include <map>
#include <vector>
#include <math.h>
#include <random>


struct DataBlock{
	std::vector<double> inputs;
	std::vector<double> ans;
};

class Layer{
	public:

		/*
		~Layer(){
			for (int i=0; i<nodesIn; i++)
				delete[] weights[i];

			delete[] weights;
			delete[] biases;
		}
		*/
		void setLayer(int a, int b) {
			srand(time(NULL));
			nodesOut = b;
			nodesIn = a;

			weights = new double*[nodesIn];
			for (int i=0; i<nodesIn; i++){
				weights[i] = new double[nodesOut];
				for(int j=0; j<nodesOut; j++)
					weights[i][j] = static_cast<double>(1 -(rand() % 2)) / sqrt(double(nodesIn)) ;
			}
			
			gradientW= new double*[nodesIn];
			for (int i=0; i<nodesIn; i++){
				gradientW[i] = new double[nodesOut];
				for(int j=0; j<nodesOut; j++)
					gradientW[i][j] = 0;
			}

			biases = new double[nodesOut];
			for(int i=0; i<nodesOut; i++) biases[i] = 0;

			gradientB= new double[nodesOut];
			for(int i=0; i<nodesOut; i++) gradientB[i] = 0;
		}

		void applyGradients(double learnRate){
			for(int nodeOut = 0; nodeOut < nodesOut; nodeOut++){
				biases[nodeOut] -= gradientB[nodeOut] * learnRate;
				for(int nodeIn=0; nodeIn < nodesIn; nodeIn++)
					weights[nodeIn][nodeOut] -= gradientW[nodeIn][nodeOut] * learnRate;
			}
		}
		
		std::vector<double> Output(std::vector<double> input){
			std::vector<double> activations(nodesOut);
			std::vector<double> wInputs(nodesOut);

			for(int nodeOut = 0; nodeOut < nodesOut; nodeOut++){
				double wInput = biases[nodeOut];
				for(int nodeIn=0; nodeIn < nodesIn; nodeIn++){
					wInput += input[nodeIn] * weights[nodeIn][nodeOut];
				}
				wInputs[nodeOut] = wInput;
				activations[nodeOut] = activation(wInput);
			}

			activationsG = activations;
			wInputsG = wInputs;
			InputsG = input;

			return activations;
		}

			/* Sigmoid */
		double activation(double wInput){
			return 1 / (1 + exp(-wInput));
		}
		double activationDer(double wInput){
			double act = activation(wInput);
			return act * (1 - act);
		}

		double nodeCost(double guess, double ans){
			double error = guess - ans;
			return error * error;

		}
		double nodeCostDer(double guess, double ans){
			return 2 * (guess - ans);

		}
		std::vector<double> calculateOutputsNV(std::vector<double> ans){
			std::vector<double> nodeVals(ans.size());
			for(int i=0; i< nodeVals.size(); i++){
				double costDer = nodeCostDer(activationsG[i], ans[i]);
				double actDer = activationDer(wInputsG[i]);
				nodeVals[i] = actDer * costDer;
			}
			return nodeVals;
		}

		void updateGradients(std::vector<double> nodeVals){
			for(int nodeOut = 0; nodeOut < nodesOut; nodeOut++){
				for(int nodeIn=0; nodeIn < nodesIn; nodeIn++){
					gradientW[nodeIn][nodeOut] += InputsG[nodeIn] * nodeVals[nodeOut];
				}
				gradientB[nodeOut] += 1 * nodeVals[nodeOut];
			}

			
		}
		void resetGradients(){
			for(int nodeOut = 0; nodeOut < nodesOut; nodeOut++){
				for(int nodeIn=0; nodeIn < nodesIn; nodeIn++){
					gradientW[nodeIn][nodeOut] = 0;
				}
				gradientB[nodeOut] = 0;
			}

			
		}

		std::vector<double> calculateHiddenNodeVals(Layer oldLayer, std::vector<double> oldNodeVals){
			std::vector<double> newNodeVals(nodesOut);

			for(int newNodeInd=0; newNodeInd < newNodeVals.size(); newNodeInd++){
				double newNodeVal=0;
				for(int oldNodeInd=0; oldNodeInd < oldNodeVals.size(); oldNodeInd++){
					newNodeVal = oldLayer.weights[newNodeInd][oldNodeInd] * oldNodeVals[oldNodeInd];
				}
				newNodeVal *= activationDer(wInputsG[newNodeInd]);
				newNodeVals[newNodeInd] = newNodeVal;
			}
			return newNodeVals;
		}


		int nodesOut=0, nodesIn=0;
		double** weights;
		double* biases;

		double** gradientW;
		double* gradientB;
		std::vector<double> activationsG;
		std::vector<double> wInputsG;
		std::vector<double> InputsG;
};

class Network{
	public:
		Network(std::vector<int> layersSizes){
			layers.resize(layersSizes.size()-1);
			for(int i=0; i < layersSizes.size() - 1; i++)
				layers[i].setLayer(layersSizes[i], layersSizes[i + 1]);
		}

		std::vector<double> calculateOutputs(std::vector<double> inputs){
			for(auto &layer : layers){
				inputs = layer.Output(inputs);

			}
			return inputs;
		}

		int classify(std::vector<double> inputs){
			std::vector<double> out = calculateOutputs(inputs);
			int ind = 0;
			double maxVal = out[0];
			for(auto &node : out){
				if(node > maxVal){
					maxVal = node;
					ind++;
				}
			}

			return ind;
			
		}
		double loss(DataBlock dataBlock){
			std::vector<double> out = calculateOutputs(dataBlock.inputs);
			Layer outLayer = layers[layers.size() - 1];
			double cost = 0;

			for(int nodeOut=0; nodeOut < out.size(); nodeOut++)
				cost += outLayer.nodeCost(out[nodeOut], dataBlock.ans[nodeOut]);
			return cost;
		}

		double loss(std::vector<DataBlock> data){
			double totalCost=0;
			for(auto &dataBlock : data){
				totalCost += loss(dataBlock);
			}
			return totalCost / data.size();
		}
		
		/*
		void Learn(std::vector<DataBlock> data, double learnRate){
			const double h = 0.0001;
			double originalLoss = loss(data);

			for(auto &layer : layers){
				for(int nodeIn=0; nodeIn < layer.nodesIn; nodeIn++){

					for(int nodeOut=0; nodeOut < layer.nodesOut; nodeOut++){
						layer.weights[nodeIn][nodeOut] += h;
						double delta = loss(data) - originalLoss;
						layer.weights[nodeIn][nodeOut] -= h;
						layer.gradientW[nodeIn][nodeOut] = delta / h;
					}
				}

				for(int biasInd=0; biasInd < layer.nodesOut; biasInd++){
					layer.biases[biasInd] += h;
					double delta = loss(data) - originalLoss;
					layer.biases[biasInd] -= h;
					layer.gradientB[biasInd] = delta / h;
				}

			}
			for(auto &layer : layers){
				layer.applyGradients(learnRate);
			}
		
		}
		*/

		void Learn(std::vector<DataBlock> data, double learnRate){
			for(auto &dataBlock : data){
				updateGradients(dataBlock);
			}
			for(auto &layer : layers){
				layer.applyGradients(learnRate);///data.size());
			}
			for(auto &layer : layers){
				layer.resetGradients();
			}
		}

			/* back propagation */
		void updateGradients(DataBlock dataBlock){
			calculateOutputs(dataBlock.inputs);

			Layer outLayer = layers[layers.size() - 1];
			std::vector<double> nodeVals = outLayer.calculateOutputsNV(dataBlock.ans);
			outLayer.updateGradients(nodeVals);

			for(int i=layers.size() - 2; i >= 0; i--){
				Layer hlayer = layers[i];
				nodeVals = hlayer.calculateHiddenNodeVals(layers[i + 1],nodeVals);
				hlayer.updateGradients(nodeVals);

			}
		}

		void printLayers(){
    			int ctr=0;
    			for(auto &layer : layers){

    				for(int i=0; i < layer.nodesIn; i++){
    					std::cout << "W("<<ctr++<<") ";
    					for(int j=0; j < layer.nodesOut; j++){
    						std::cout <<layer.weights[i][j] << " | ";
    					}
    					std::cout << "\t";
    				}
    				std::cout << '\n';

    			}
    		}

	private:
		std::vector<Layer> layers;

};
