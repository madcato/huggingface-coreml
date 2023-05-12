//
//  DistilGPT2TokenizerTests.swift
//  huggingface-ios-appTests
//
//  Created by Daniel Vela on 10/5/23.
//

@testable import huggingface_ios_app
import XCTest
import CoreML

final class DistilGPT2TokenizerTests: XCTestCase {

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    func testCreateOk() throws {
        do {
            let sut = try DistilGPT2Tokenizer.from_pretrained("distilgpt2")
            XCTAssertNotNil(sut)
        } catch {
            XCTAssertTrue(false)
        }
    }
    
    func testCreateError() throws {
        do {
            let _ = try DistilGPT2Tokenizer.from_pretrained("distil")
        } catch {
            XCTAssertNotNil(error)
        }
    }

    func testDecodeDistilgpt2() {
        do {
            let sut = try DistilGPT2Tokenizer.from_pretrained("distilgpt2")
            let tokens = sut.encode(text: "Hello world")  // Put a white space always frist position
            XCTAssertEqual(tokens[0], 15496)
            XCTAssertEqual(tokens[1], 995)
            let untokenizedStr = sut.decode(tokens: tokens)
            XCTAssertEqual(untokenizedStr, "Hello world")
        } catch {
            XCTAssertTrue(false)
        }
    }
    
    func testModelPredict() {
        do {
            let sut = try DistilGPT2Tokenizer.from_pretrained("distilgpt2")
            let tokens = sut.encode(text: " Hello world how are you from here to there")  // Put a white space always frist position
            
            let seqLen = 128
            
            let maxTokens = (tokens.count > seqLen)
                ? Array(tokens[..<seqLen])
                : tokens
            
            /// Pad input_ids on the right, up to `seqLen`:
            let input_ids = MLMultiArray.from(
                maxTokens + Array(repeating: 0, count: seqLen - maxTokens.count)
                , dims: 2
            )
            
            
            let masks = MLMultiArray.from(
                Array(repeating: 1, count: seqLen)
                , dims: 2
            )
            
            let input = distilgpt2Input(input_ids: input_ids, attention_mask: masks)
            
            let model = try distilgpt2()
            
            let output: distilgpt2Output = try model.prediction(input: input)
            
            let outputLogits = MLMultiArray.slice(
                output.token_scores,
                indexing: [.select(0), .select(tokens.count - 1), .slice]
            )

            let nextToken = Math.argmax(outputLogits)
            let tok = nextToken.0
            let nextWord = sut.decode(tokens: [tok])
            
            let a = 0 + 4
            
            print("hola")
//            let untokenizedStr = sut.decode(tokens: tokenIds)
//            XCTAssertEqual(untokenizedStr, " Hello world")
        } catch {
            print(error.localizedDescription)
            XCTAssertTrue(false)
        }
    }
}
