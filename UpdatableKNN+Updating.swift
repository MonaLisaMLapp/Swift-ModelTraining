//
//  UpdatableKNN+Updating.swift
//  MonaLisa
//
//  Created by Raphael Ferezin Kitahara on 07/08/24.
//

import CoreML

extension UpdatableKNN {
    /// Creates an update model from a given model URL and training data.
    ///
    /// - Parameters:
    ///     - url: The location of the model the Update Task will update.
    ///     - trainingData: The training data the Update Task uses to update the model.
    ///     - completionHandler: A closure the Update Task calls when it finishes updating the model.
    /// - Tag: CreateUpdateTask
    static func updateModel(at url: URL,
                            input: String,
                            label: String,
                            completionHandler: @escaping (MLUpdateContext) -> Void) {
        
        
        var featureProviders = [MLFeatureProvider]()
        
        let inputName = "input" // Replace with your actual input feature name
        let outputName = "output" // Replace with your actual output label name
        
        guard let multiArray = textToMLMultiArray(text: input) else {
            return
        }
        
        let inputValue = MLFeatureValue(multiArray: multiArray)
        let outputValue = MLFeatureValue(string: label)
        
        let dataPointFeatures: [String: MLFeatureValue] = [inputName: inputValue,
                                                           outputName: outputValue]
        
        if let provider = try? MLDictionaryFeatureProvider(dictionary: dataPointFeatures) {
            featureProviders.append(provider)
        }
        
        // Combine the input and output into a single training example
        let trainingData = MLArrayBatchProvider(array: featureProviders)
        
        // Create an Update Task.
        guard let updateTask = try? MLUpdateTask(forModelAt: url,
                                                 trainingData: trainingData,
                                                 configuration: nil,
                                                 completionHandler: completionHandler)
        else {
            print("Could't create an MLUpdateTask.")
            return
        }

        updateTask.resume()
    }
}
