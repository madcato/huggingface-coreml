//
//  DialogGPTModel.swift
//  huggingface-ios-app
//
//  Created by Daniel Vela on 14/5/23.
//

import Foundation
import CoreML

class DialogGPTModel {
    let model: DialoGPT_small
//    let tokenizer: BlenderbotSmallTokenizer
    let seqLen = 12
    
    let eos_token_id = 50256
    
    init() throws {
//        let modelName = "facebook/blenderbot-small-90M"
//        self.tokenizer = try BlenderbotSmallTokenizer.from_pretrained(modelName)
        self.model = try DialoGPT_small()
    }
    
    func generate(_ string: String) throws -> String {
        var tokens = [13921,  1637,  2822, 12157,    30, 50256]
        
        
        for _ in 0..<6 {
            var next_token = try predict(tokens)
            tokens.append(next_token)
            print(String(next_token) + ", ")
        }
        
        return ""
    }
    
    private func predict(_ tokens: [Int]) throws -> Int {
        var prev_prob = 0.0
        var next_token = 0
        let maxTokens = (tokens.count > seqLen)
            ? Array(tokens[..<seqLen])
            : tokens

        let inputArray = MLMultiArray.from(maxTokens + Array(repeating: eos_token_id, count: seqLen - maxTokens.count), dims: 2)
        let maskArray = MLMultiArray.from(Array(repeating: 1, count: maxTokens.count) + Array(repeating: 0, count: seqLen - maxTokens.count)
            , dims: 2
        )
        let inputEncoder = DialoGPT_smallInput(input_ids: inputArray, attention_mask: maskArray)
        
        let output = try model.prediction(input: inputEncoder)
        
        
        for j in 0..<output.token_scores.shape[1].intValue {
            let outputLogits = MLMultiArray.slice(
                output.token_scores,
                indexing: [.select(0), .select(j), .slice]
            )

            let nextToken = Math.argmax(outputLogits)
            let tok = nextToken.0
            let prob = nextToken.1
            
            if prob > prev_prob {
                next_token = tok
                prev_prob = prob
            }

        }

        return next_token
    }
}
