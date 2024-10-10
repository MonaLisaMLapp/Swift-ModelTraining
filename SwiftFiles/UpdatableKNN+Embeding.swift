//
//  UpdatableKNN+Embeding.swift
//  MonaLisa
//
//  Created by Pedro Ceccon on 20/08/24.
//

import Foundation
import CoreML
import NaturalLanguage

extension UpdatableKNN {
    // Function to convert a text string to MLMultiArray
    static func textToMLMultiArray(text: String) -> MLMultiArray? {
        // Load the English word embedding model
        guard let embedding = NLEmbedding.wordEmbedding(for: .english) else {
            print("Failed to load word embedding model.")
            return nil
        }
        
        // Split the text into words and get their embeddings
        let words = text.lowercased().split(separator: " ")
        var embeddingVectors: [[Float32]] = []
        
        for word in words {
            if let vector = embedding.vector(for: String(word)) {
                let floatVector = vector.map { Float32($0) }
                embeddingVectors.append(floatVector)
            }
        }
        
        // Average the word vectors to get a single embedding for the whole text
        guard !embeddingVectors.isEmpty else {
            print("No embeddings found for the input text.")
            return nil
        }
        
        let vectorLength = embeddingVectors[0].count
        var averagedVector = [Float32](repeating: 0, count: vectorLength)
        
        for vector in embeddingVectors {
            for i in 0..<vectorLength {
                averagedVector[i] += vector[i]
            }
        }
        
        for i in 0..<vectorLength {
            averagedVector[i] /= Float32(embeddingVectors.count)
        }
        
        // Ensure the vector has 128 dimensions (truncate or pad with zeros if necessary)
        let targetDimension = 128
        if averagedVector.count > targetDimension {
            averagedVector = Array(averagedVector.prefix(targetDimension))
        } else if averagedVector.count < targetDimension {
            averagedVector.append(contentsOf: [Float32](repeating: 0, count: targetDimension - averagedVector.count))
        }
        
        // Convert the averaged vector to MLMultiArray
        do {
            let multiArray = try MLMultiArray(shape: [targetDimension as NSNumber], dataType: .float32)
            for (index, value) in averagedVector.enumerated() {
                multiArray[index] = NSNumber(value: value)
            }
            return multiArray
        } catch {
            print("Error creating MLMultiArray: \(error)")
            return nil
        }
    }
}
