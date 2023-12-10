const mongoose = require('mongoose')
const place_type = require('./PlaceType')

const NearbySchema = new mongoose.Schema({
    location: {
        type: String,
        required: true
    },
    placetype: {
        type: String,
        enum: place_type,
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

module.exports = mongoose.model('Nearby', NearbySchema)