//
//  T5SmallModel.swift
//  huggingface-ios-app
//
//  Created by Daniel Vela on 13/5/23.
//

import Foundation
import CoreML

class T5SmallModel {
    let encoder: encoder_t5_small
    let decoder: decoder_t5_small
//    let tokenizer: BlenderbotSmallTokenizer
    let seqLen = 64
    
    init() throws {
//        let modelName = "facebook/blenderbot-small-90M"
//        self.tokenizer = try BlenderbotSmallTokenizer.from_pretrained(modelName)
        self.encoder = try encoder_t5_small()
        self.decoder = try decoder_t5_small()
    }
    
    enum DecodingStrategy {
        /// At each time step, we select the most likely next token
        case greedy
        /// Sample only from the top-k most-probable tokens (k is a hyper-parameter).
        case topK(Int)
        /// Sample from the top tokens with a cumulative probability just above a threshold (nucleus/top-p).
        case topP(Double)
    }
    
    private let strategy = DecodingStrategy.greedy
    
    func predict(_ tokens: [Int]) throws -> Int {
//        let tokens = tokenizer.encode(text: string)
//        let tokens = [1] + tokens_0 + [2]
//        let tokens = [8774,  296,  149,   33,   25, 58,   1, 0 , 0, 0, 0, 0, 0,0, 0,0 ,0, 0,0 ,0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0]
        

        let maxTokens = (tokens.count > seqLen)
            ? Array(tokens[..<seqLen])
            : tokens

        let input_ids = MLMultiArray.from(
            maxTokens + Array(repeating: 0, count: seqLen - maxTokens.count), dims: 2
        )
        let position_ids = MLMultiArray.from(
            Array(repeating: 1, count: maxTokens.count) + Array(repeating: 0, count: seqLen - maxTokens.count ), dims: 2
        )
        
        let outputEncoder = try encoder.prediction(input_ids: input_ids, attention_mask: position_ids)
        
        let outputDecoder = try decoder.prediction(decoder_input_ids: input_ids, decoder_attention_mask: position_ids, encoder_last_hidden_state: outputEncoder.last_hidden_state, encoder_attention_mask: position_ids)
        
        
        let outputLogits = MLMultiArray.slice(
            outputDecoder.token_scores,
            indexing: [.select(0), .select(maxTokens.count - 1), .slice]
        )
        switch strategy {
        case .greedy:
            let nextToken = Math.argmax(outputLogits)
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
    
    // Main generation loop.
    ///
    /// Will generate next `nTokens` (defaults to 10).
    /// Calls an incremental `callback` for each new token, then returns the generated string at the end.
    ///
    func generate(text: String, nTokens: Int = 30) throws -> String {
//        var tokens = tokenizer.encode(text: text)
        var tokens = [363,  31,   7,  39, 564,  58,   1]
        var newTokens: [Int] = []
        for i in 0..<nTokens {
            let nextToken = try predict(tokens)

            tokens.append(nextToken)
            newTokens.append(nextToken)
            print("🦄 ", i, nextToken, tokens.count)
            
        }
        return ""  // tokenizer.decode(tokens: newTokens)
    }
}
