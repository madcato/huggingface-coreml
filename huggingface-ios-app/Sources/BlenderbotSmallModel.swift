//
//  BlenderbotSmallModel.swift
//  huggingface-ios-app
//
//  Created by Daniel Vela on 11/5/23.
//

import Foundation
import CoreML

class BlenderbotSmallModel {
    let encoder: blenderbot_small_90M_encoder
    let decoder: blenderbot_small_90M_decoder
    let tokenizer: BlenderbotSmallTokenizer
    let seqLen = 20
    
    init() throws {
        let modelName = "facebook/blenderbot-small-90M"
        self.tokenizer = try BlenderbotSmallTokenizer.from_pretrained(modelName)
        self.encoder = try blenderbot_small_90M_encoder()
        self.decoder = try blenderbot_small_90M_decoder()
    }
    
    func predict(_ string: String) throws -> String {
        let tokens_0 = tokenizer.encode(text: string)
        let tokens = [1] + tokens_0 + [2]
        
        let maxTokens = (tokens.count > seqLen)
            ? Array(tokens[..<seqLen])
            : tokens
        
        let inputArray = MLMultiArray.from(maxTokens + Array(repeating: 0, count: seqLen - maxTokens.count), dims: 2)
        let maskArray = MLMultiArray.from(
            Array(repeating: 1, count: seqLen)
            , dims: 2
        )
        let inputEncoder = blenderbot_small_90M_encoderInput(input_ids: inputArray , attention_mask: maskArray)
        
        let outputEncoder = try encoder.prediction(input: inputEncoder)
        
        let outputDecoder = try decoder.prediction(decoder_input_ids: inputArray, decoder_attention_mask: maskArray, encoder_last_hidden_state: outputEncoder.last_hidden_state, encoder_attention_mask: maskArray)
        
        let a = outputDecoder.token_scores
        
        
        return ""
    }
}
