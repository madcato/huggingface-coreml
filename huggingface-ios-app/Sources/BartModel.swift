//
//  BartModel.swift
//  huggingface-ios-app
//
//  Created by Daniel Vela on 12/5/23.
//

import Foundation
import CoreML

class BartModel {
    let encoder: bart_large_encoder
//    let decoder: bart_large_decoder
    let tokenizer: DistilGPT2Tokenizer
    let seqLen = 128
    
    init() throws {
        let modelName = "facebook/bart-large"
        self.tokenizer = try DistilGPT2Tokenizer.from_pretrained(modelName)
        self.encoder = try bart_large_encoder()
//        self.decoder = try bart_large_decoder()
    }
    
    func predict(_ string: String) throws -> String {
        let tokens = tokenizer.encode(text: string)
        
        let maxTokens = (tokens.count > seqLen)
            ? Array(tokens[..<seqLen])
            : tokens
        
        let inputArray = MLMultiArray.from(maxTokens + Array(repeating: 0, count: seqLen - maxTokens.count), dims: 2)
        let maskArray = MLMultiArray.from(
            Array(repeating: 1, count: seqLen)
            , dims: 2
        )
        let inputEncoder = bart_large_encoderInput(input_ids: inputArray , attention_mask: maskArray)
        
        let outputEncoder = try encoder.prediction(input: inputEncoder)
        
//        let outputDecoder = try decoder.prediction(decoder_input_ids: inputArray, decoder_attention_mask: maskArray, encoder_last_hidden_state: outputEncoder.last_hidden_state, encoder_attention_mask: maskArray)
//
//        let a = outputDecoder.token_scores
        
        
        return ""
    }
}
