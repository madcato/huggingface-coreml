//
//  GPT2Model.swift
//  huggingface-ios-app
//
//  Created by Daniel Vela on 14/5/23.
//

import Foundation
import CoreML

final class GPT2Model {
    private let model: DialoGPT_small
    private let tokenizer: GPT2Tokenizer
    private let seqLen = 24
    
    init() throws {
        let modelName = "distilgpt2"
        self.tokenizer = try GPT2Tokenizer.from_pretrained(modelName)
        self.model = try DialoGPT_small()
    }

    enum DecodingStrategy {
        /// At each time step, we select the most likely next token
        case greedy
        /// Sample only from the top-k most-probable tokens (k is a hyper-parameter).
        case topK(Int)
        /// Sample from the top tokens with a cumulative probability just above a threshold (nucleus/top-p).
        case topP(Double)
    }

    private let strategy = DecodingStrategy.topK(24)
    
    /// Main prediction loop:
    /// Predict next token from array of previous tokens.
    /// - featurization
    /// - model inference
    /// - Decoding according to the model's `strategy`
    func predict(tokens: [Int]) -> Int {
        let maxTokens = (tokens.count > seqLen)
            ? Array(tokens[..<seqLen])
            : tokens
        
        /// Pad input_ids on the right, up to `seqLen`:
        let input_ids = MLMultiArray.from(
            maxTokens + Array(repeating: 0, count: seqLen - maxTokens.count), dims: 2
        )
        let maskArray = MLMultiArray.from(
            Array(repeating: 1, count: maxTokens.count) + Array(repeating: 0, count: seqLen - maxTokens.count)
            , dims: 2
        )
        let input = DialoGPT_smallInput(input_ids: input_ids, attention_mask: maskArray)
        let output = try! model.prediction(input: input)
        
        for i in 0..<output.token_scores.shape[1].intValue {
            let outputLogits = MLMultiArray.slice(
                output.token_scores,
                indexing: [.select(0), .select(i), .slice]
            )
            
            switch strategy {
            case .greedy:
                let nextToken = Math.argmax(outputLogits)
                print(nextToken.0)
                return nextToken.0
            case .topK(let k):
                let logits = MLMultiArray.toDoubleArray(outputLogits)
                let topk = Math.topK(arr: logits, k: k)
                let sampleIndex = Math.sample(indexes: topk.indexes, probs: topk.probs)
                return sampleIndex
            case .topP(_):
                fatalError("topP is not implemented yet")
            }
        }
        return 0
    }
    
    
    /// Main generation loop.
    ///
    /// Will generate next `nTokens` (defaults to 10).
    /// Calls an incremental `callback` for each new token, then returns the generated string at the end.
    ///
    func generate(text: String, nTokens: Int = 10, callback: ((String, Double) -> Void)?) -> String {
        var tokens = tokenizer.encode(text: text)
        var newTokens: [Int] = []
        for i in 0..<nTokens {
            let (nextToken, time) = Utils.time {
                return predict(tokens: tokens)
            }
            
            tokens.append(nextToken)
            newTokens.append(nextToken)
            print("🦄 <\(time)s>", i, nextToken, tokens.count)
            callback?(
                tokenizer.decode(tokens: newTokens), time
            )
        }
        return tokenizer.decode(tokens: newTokens)
    }
}
