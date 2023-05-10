//
//  BlenderbotTokenizerTests.swift
//  huggingface-ios-appTests
//
//  Created by Daniel Vela on 10/5/23.
//

@testable import huggingface_ios_app
import XCTest
import CoreML

final class BlenderbotTokenizerTests: XCTestCase {

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    func testCreateOk() throws {
        do {
            let sut = try BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
            XCTAssertNotNil(sut)
        } catch {
            XCTAssertTrue(false)
        }
    }
    
    func testCreateError() throws {
        do {
            let _ = try BlenderbotTokenizer.from_pretrained("facebook/blenderbot-40")
        } catch {
            XCTAssertNotNil(error)
        }
    }

    func testDecodeFaceBookBlenderbot400MDistill() {
        do {
            let sut = try BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
            let tokens = sut.encode(text: " Hello world")  // Put a white space always frist position
//            XCTAssertEqual(tokens[0], 6950)
//            XCTAssertEqual(tokens[1], 1085)
//            XCTAssertEqual(tokens[2], 2)
            let untokenizedStr = sut.decode(tokens: tokens)
            XCTAssertEqual(untokenizedStr, " Hello world")
        } catch {
            XCTAssertTrue(false)
        }
    }

    func testModelPredict() {
        do {
            let sut = try BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
//            let tokens = sut.encode(text: " Hello world how are you from here to there")  // Put a white space always frist position
//            XCTAssertEqual(tokens[0], 6950)
//            XCTAssertEqual(tokens[1], 1085)
//            XCTAssertEqual(tokens[2], 2)

            let tokens = [6950, 1085, 2]
            let masks = tokens.map { _ in 1 }

            let input = blenderbot_400M_distillInput(input_ids: MLMultiArray.from(tokens, dims: 2), attention_mask: MLMultiArray.from(masks, dims: 2))

            let model = try blenderbot_400M_distill()

            let output: blenderbot_400M_distillOutput = try model.prediction(input: input)
//            let tokenIds = Array(MLMultiArray.toDoubleArray(output.last_hidden_state))

            let outputLogits = MLMultiArray.slice(
                output.last_hidden_state,
                indexing: [.select(0), .select(tokens.count - 1), .slice]
            )

            let nextToken = Math.argmax(outputLogits)
            let tok = nextToken.0

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
