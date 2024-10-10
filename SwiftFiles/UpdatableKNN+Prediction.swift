//
//  UpdatableKNN+Prediction.swift
//  MonaLisa
//
//  Created by Raphael Ferezin Kitahara on 07/08/24.
//

import Foundation
import CoreML
import NaturalLanguage

extension UpdatableKNN {
    static let unknownLabel = "Other"

    /// Predicts a label for the given transaction description.
    /// - Parameter value: A user's transaction description represented as a feature value.
    /// - Returns: The predicted string label, if known; otherwise `nil`.
    func predictLabelFor(_ input: String) -> String? {

        guard let inputValue = UpdatableKNN.textToMLMultiArray(text: input) else {
            return nil
        }
        guard let prediction = try? prediction(input: UpdatableKNNInput(input: inputValue)).outputProbs else {
            return nil
        }
        
        var label: String? = nil
        
        for (key, value) in prediction {
            print("\t" + key + ": " + String(value))
            if value > 0.5 {
                label = key
            }
        }
        
        return label
    }
}
