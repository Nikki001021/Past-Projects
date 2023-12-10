const mongoose = require('mongoose')
const states = require('./State')

const EntrySchema = new mongoose.Schema({
    location: {
        type: String,
        required: true
    },
    minRent: {
        type: String,
        required: true
    },
    maxRent: {
        type: String,
        required: true
    },
    minbedrooms: {
        type: String,
        enum: ['0', '1', '2', '3', '4', '5', '6'],
        required: true
    },
    maxbedrooms: {
        type: String,
        enum: ['0', '1', '2', '3', '4', '5', '6'],
        required: true
    },
    minbathrooms: {
        type: String,
        enum: ['0', '1', '2', '3', '4', '5', '6'],
        required: true
    },
    maxbathrooms: {
        type: String,
        enum: ['0', '1', '2', '3', '4', '5', '6'],
        required: true
    },
    user: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
})

module.exports = mongoose.model('Entry', EntrySchema)