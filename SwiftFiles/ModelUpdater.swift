//
//  ModelUpdater.swift
//  MonaLisa
//
//  Created by Raphael Ferezin Kitahara on 07/08/24.
//

import CoreML

/// Class that handles predictions and updating of UpdatableTransactionClassifier model.
struct ModelUpdater {
    // MARK: - Private Type Properties
    /// The updated Transaction Classifier model.
    private static var updatedKNNClassifier: UpdatableKNN?
    /// The default Transaction Classifier model.
    private static var defaultKNNClassifier: UpdatableKNN {
        do {
            return try UpdatableKNN(configuration: .init())
        } catch {
            fatalError("Couldn't load UpdatableTransactionClassifier due to: \(error.localizedDescription)")
        }
    }
    
    static func exportModel() {
        do {
                // Load the compiled model from the URL
                let compiledModelDirectory = updatedModelURL.deletingLastPathComponent()
                
                // Specify the output URL for the mlmodel file
                let documentDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                let outputURL = documentDirectory.appendingPathComponent("UpdatedModel.mlmodel")
                
                // Compile the mlmodelc back to an mlmodel
                try FileManager.default.copyItem(at: compiledModelDirectory, to: outputURL)
                
                print("Model exported successfully to \(outputURL)")
            } catch {
                print("Failed to export the model: \(error.localizedDescription)")
            }
    }
    
    // The Transaction Classifier model currently in use.
    private static var liveModel: UpdatableKNN {
        updatedKNNClassifier ?? defaultKNNClassifier
    }
    
    /// The location of the app's Application Support directory for the user.
    private static let appDirectory = FileManager.default.urls(for: .applicationSupportDirectory,
                                                               in: .userDomainMask).first!
    
    /// The default Transaction Classifier model's file URL.
    private static let defaultModelURL = UpdatableKNN.urlOfModelInThisBundle
    /// The permanent location of the updated Transaction Classifier model.
    private static var updatedModelURL = appDirectory.appendingPathComponent("personalized.mlmodelc")
    /// The temporary location of the updated Transaction Classifier model.
    private static var tempUpdatedModelURL = appDirectory.appendingPathComponent("personalized_tmp.mlmodelc")
    
    /// Triggers code on the first prediction, to (potentially) load a previously saved updated model just-in-time.
    private static var hasMadeFirstPrediction = false
    
    /// The Model Updater type doesn't use instances of itself.
    private init() { }
    
    // MARK: - Public Type Methods
    static func predictLabelFor(_ value: String) -> String? {
        if (!hasMadeFirstPrediction) {
            hasMadeFirstPrediction = true
            
            // Load the updated model the app saved on an earlier run, if available.
            loadUpdatedModel()
        }
        
        return liveModel.predictLabelFor(value)
    }
    
    /// Updates the model to recognize transactions similar to the given descriptions contained within the `inputBatchProvider`.
    /// - Parameters:
    ///     - trainingData: A collection of sample texts, each paired with the same label.
    ///     - completionHandler: The completion handler provided from a view controller.
    /// - Tag: CreateUpdateTask
    static func updateWith(input: String,
                           label: String,
                           completionHandler: @escaping () -> Void) {
        
        /// The URL of the currently active Transaction Classifier.
        let usingUpdatedModel = updatedKNNClassifier != nil
        let currentModelURL = usingUpdatedModel ? updatedModelURL : defaultModelURL
        
        /// The closure an MLUpdateTask calls when it finishes updating the model.
        func updateModelCompletionHandler(updateContext: MLUpdateContext) {
            // Save the updated model to the file system.
            saveUpdatedModel(updateContext)
            
            // Begin using the saved updated model.
            loadUpdatedModel()
            
            // Inform the calling View Controller when the update is complete
            DispatchQueue.main.async { completionHandler() }
        }
        
        UpdatableKNN.updateModel(at: currentModelURL,
                                 input: input,
                                 label: label,
                                 completionHandler: updateModelCompletionHandler)
    }
    
    /// Deletes the updated model and reverts back to the original Transaction Classifier.
    static func resetTransactionClassifier() {
        // Clear the updated Transaction Classifier.
        updatedKNNClassifier = nil
        
        // Remove the updated model from its designated path.
        if (FileManager.default.fileExists(atPath: updatedModelURL.path)) {
            try? FileManager.default.removeItem(at: updatedModelURL)
        }
    }
    
    // MARK: - Private Type Helper Methods
    /// Saves the model in the given Update Context provided by an MLUpdateTask.
    /// - Parameter updateContext: The context from the Update Task that contains the updated model.
    /// - Tag: SaveUpdatedModel
    private static func saveUpdatedModel(_ updateContext: MLUpdateContext) {
        let updatedModel = updateContext.model
        let fileManager = FileManager.default
        do {
            // Create a directory for the updated model.
            try fileManager.createDirectory(at: tempUpdatedModelURL,
                                            withIntermediateDirectories: true,
                                            attributes: nil)
            
            // Save the updated model to a temporary filename.
            try updatedModel.write(to: tempUpdatedModelURL)
            
            // Replace any previously updated model with this one.
            _ = try fileManager.replaceItemAt(updatedModelURL,
                                              withItemAt: tempUpdatedModelURL)
            
            print("Updated model saved to:\n\t\(updatedModelURL)")
        } catch let error {
            print("Could not save updated model to the file system: \(error)")
            return
        }
    }
    
    /// Loads the updated Transaction Classifier, if available.
    /// - Tag: LoadUpdatedModel
    private static func loadUpdatedModel() {
        guard FileManager.default.fileExists(atPath: updatedModelURL.path) else {
            // The updated model is not present at its designated path.
            return
        }
        
        // Create an instance of the updated model.
        guard let model = try? UpdatableKNN(contentsOf: updatedModelURL) else {
            return
        }
        
        // Use this updated model to make predictions in the future.
        updatedKNNClassifier = model
    }
}
