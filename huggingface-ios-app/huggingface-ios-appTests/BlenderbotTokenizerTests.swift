//
//  BlenderbotTokenizerTests.swift
//  huggingface-ios-appTests
//
//  Created by Daniel Vela on 10/5/23.
//

@testable import huggingface_ios_app
import XCTest

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
            XCTAssertEqual(tokens[0], 6950)
            XCTAssertEqual(tokens[1], 1085)
            XCTAssertEqual(tokens[2], 2)
            let untokenizedStr = sut.decode(tokens: tokens)
            XCTAssertEqual(untokenizedStr, " Hello world")
        } catch {
            XCTAssertTrue(false)
        }
    }
}
