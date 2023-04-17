package com.example.easychatgpt

class Message(var message: String, var sentBy: String) {

    companion object {
        @JvmField
        var SENT_BY_ME = "me"
        var SENT_BY_BOT = "bot"
    }
}